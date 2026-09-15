# -*- coding: utf-8 -*-
"""تست‌های پذیرشِ فاز ۲d — پیش‌ثبت، آزمونِ منجمد و داوریِ یک‌بارمصرف.

معیارها (بخشِ ۶ پلن):
* روی دادهٔ بدونِ لبه، داور هیچ ترکیبی را قبول نمی‌کند.
* با لبهٔ واقعی و نمونهٔ کافی، قبول می‌کند.
* قضاوتِ دوباره استثنا می‌دهد.
* ویرایشِ پیش‌ثبت پس از ثبت، داور را متوقف می‌کند.
* فقط داور می‌تواند فهرستِ مجاز را بنویسد، و پیش‌ثبتِ تازه آن را صفر می‌کند.
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import calib  # noqa: E402
import gates  # noqa: E402
import research  # noqa: E402

H4 = research.TF_MS["4h"]
D1 = research.TF_MS["1d"]
START = 1_600_000_000_000
NOW = START / 1000 + 5 * 365 * 86400          # ثبت پنج سال پس از شروعِ داده


def events_for(tf, setup, side, n, edge, rng, symbols, spacing_bars=12):
    """رویدادهای مصنوعیِ یک فرضیه در سراسرِ بازه، با ساختارِ 1.8R/−1R."""
    step = research.TF_MS[tf] * float(spacing_bars)
    out = []
    base = rng.random(n)
    r = np.where(base < 0.643 - edge / 2.8, -1.0, 1.8)
    for k in range(n):
        out.append({
            "ts": int(START + k * step),
            "sym": symbols[k % len(symbols)],
            "setup": setup, "dir": 1 if side == "long" else -1,
            "r": float(r[k]), "risk_pct": 1.0, "cost_pct": 0.0,
        })
    return out


class _ResearchMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = (research.PREREG_PATH, research.FINAL_DIR, gates.GATES_PATH)
        research.PREREG_PATH = os.path.join(self.tmp.name, "preregistration.json")
        research.FINAL_DIR = os.path.join(self.tmp.name, "final_test")
        gates.GATES_PATH = os.path.join(self.tmp.name, "gates.json")
        with open(gates.GATES_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(gates.DEFAULTS), f)
        gates.load_gates(force=True)

    def tearDown(self):
        research.PREREG_PATH, research.FINAL_DIR, gates.GATES_PATH = self._old
        gates._cache.update(mtime=None, data=None)
        self.tmp.cleanup()

    def spans(self):
        end = int(NOW * 1000)
        return {"4h": (START, end), "1d": (START, end)}

    def family_events(self, edges, n=4000, seed=5):
        rng = np.random.default_rng(seed)
        majors = list(research.MAJORS)
        by_tf = {"4h": [], "1d": []}
        for hyp in research.DEFAULT_FAMILY:
            key = gates.combo_key(hyp["tf"], hyp["setup"], hyp["side"])
            # فاصله طوری که کلِ بازهٔ پنج‌ساله پوشش داده شود (اعشاری — گردکردن بازه را کوتاه می‌کرد)
            spacing = (NOW * 1000 - START) / research.TF_MS[hyp["tf"]] / n
            by_tf[hyp["tf"]].extend(events_for(hyp["tf"], hyp["setup"], hyp["side"], n,
                                               edges.get(key, 0.0), rng, majors, spacing))
        return by_tf


class PreregistrationTests(_ResearchMixin, unittest.TestCase):
    def test_registration_binds_its_hash_into_gates(self):
        doc = research.register(self.spans(), now=NOW)
        self.assertEqual(gates.load_gates(force=True)["preregistration_hash"], doc["hash"])
        research.verify_prereg()

    def test_editing_the_preregistration_after_the_fact_stops_the_judge(self):
        research.register(self.spans(), now=NOW)
        doc = research.load_prereg()
        doc["gates"]["min_mean_net_r"] = 0.0           # «بعد از دیدن نتیجه آستانه را پایین بیاوریم»
        with open(research.PREREG_PATH, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        with self.assertRaises(research.PreregistrationError):
            research.judge({"4h": [], "1d": []})

    def test_each_reregistration_counts_as_another_trial(self):
        a = research.register(self.spans(), now=NOW)
        b = research.register(self.spans(), now=NOW + 1)
        self.assertEqual(a["n_trials"], 1)
        self.assertEqual(b["n_trials"], 2)
        self.assertEqual(b["previous_hash"], a["hash"])

    def test_new_registration_clears_previously_allowed_combos(self):
        research.register(self.spans(), now=NOW)
        gver = gates.load_gates(force=True)["version"]
        gates.set_allowed_combos(["4h|zx|long"], "t", judge_token=f"judge:{gver}",
                                 symbols=["BTCUSDT"])
        research.register(self.spans(), now=NOW + 1)
        self.assertEqual(gates.allowed_combos(), [])

    def test_final_window_is_the_last_quarter_up_to_registration(self):
        doc = research.register(self.spans(), now=NOW)
        w = doc["final_windows"]["4h"]
        self.assertEqual(w["end_ms"], int(NOW * 1000))
        span = int(NOW * 1000) - START
        self.assertAlmostEqual((w["end_ms"] - w["start_ms"]) / span, 0.25, places=3)
        self.assertTrue(research.in_final_window("4h", w["start_ms"], doc))
        self.assertFalse(research.in_final_window("4h", w["start_ms"] - 1, doc))
        self.assertFalse(research.in_final_window("4h", w["end_ms"], doc))

    def test_default_family_excludes_the_fee_negative_timeframes(self):
        tfs = {h["tf"] for h in research.DEFAULT_FAMILY}
        self.assertEqual(tfs, {"4h", "1d"})
        self.assertEqual(len(research.DEFAULT_FAMILY), 8)


class JudgeTests(_ResearchMixin, unittest.TestCase):
    def test_zero_edge_family_is_rejected_in_full(self):
        research.register(self.spans(), now=NOW)
        res = research.judge(self.family_events({}), now=NOW)
        self.assertEqual(res["passing"], [])
        self.assertEqual(gates.allowed_combos(), [])

    def test_a_real_edge_with_enough_data_is_accepted_and_scoped_to_majors(self):
        research.register(self.spans(), now=NOW)
        res = research.judge(self.family_events({"4h|zx|long": 0.45}, n=6000), now=NOW)
        self.assertIn("4h|zx|long", res["passing"], res["hypotheses"]["4h|zx|long"])
        self.assertEqual(gates.allowed_combos(), res["passing"])
        self.assertTrue(gates.is_combo_allowed("4h", "zx", "long", symbol="BTCUSDT"))
        self.assertFalse(gates.is_combo_allowed("4h", "zx", "long", symbol="PEPEUSDT"))

    def test_judging_twice_is_refused_even_after_a_bad_result(self):
        research.register(self.spans(), now=NOW)
        research.judge(self.family_events({}), now=NOW)
        with self.assertRaises(research.AlreadyJudged):
            research.judge(self.family_events({"4h|zx|long": 0.45}, n=6000), now=NOW)

    def test_events_outside_the_frozen_window_are_ignored(self):
        doc = research.register(self.spans(), now=NOW)
        evs = self.family_events({"4h|zx|long": 0.45}, n=6000)
        w = doc["final_windows"]["4h"]
        # فقط رویدادهای پیش از پنجره را نگه دار ⇒ داور نباید چیزی ببیند
        evs = {tf: [e for e in lst if e["ts"] < w["start_ms"]] for tf, lst in evs.items()}
        res = research.evaluate(evs, doc)
        self.assertEqual(res["hypotheses"]["4h|zx|long"]["n"], 0)

    def test_alts_do_not_count_toward_a_majors_hypothesis(self):
        doc = research.register(self.spans(), now=NOW)
        evs = self.family_events({"4h|zx|long": 0.45}, n=6000)
        for lst in evs.values():
            for e in lst:
                e["sym"] = "PEPEUSDT"
        res = research.evaluate(evs, doc)
        self.assertEqual(res["hypotheses"]["4h|zx|long"]["n"], 0)

    def test_fail_reasons_are_reported_for_every_rejected_hypothesis(self):
        doc = research.register(self.spans(), now=NOW)
        res = research.evaluate(self.family_events({}, n=400), doc)
        for key, h in res["hypotheses"].items():
            if h.get("verdict") == "fail":
                self.assertTrue(h["fail_reasons"], key)

    def test_gate_is_passable_in_principle_by_a_strong_edge(self):
        """آستانه‌ای که حتی لبهٔ قوی را رد کند، ابزارِ خراب است نه ردِ صادقانه."""
        doc = research.register(self.spans(), now=NOW)
        res = research.evaluate(self.family_events({"4h|zx|long": 0.45}, n=6000), doc)
        h = res["hypotheses"]["4h|zx|long"]
        self.assertEqual(h["evidence"], "edge_proven", h["fail_reasons"])

    def test_no_edge_and_thin_evidence_are_reported_differently(self):
        doc = research.register(self.spans(), now=NOW)
        thin = research.evaluate(self.family_events({"4h|zx|long": 0.45}, n=120), doc)
        self.assertEqual(thin["hypotheses"]["4h|zx|long"]["evidence"], "insufficient_evidence")
        flat = research.evaluate(self.family_events({"4h|zx|short": -0.4}, n=6000), doc)
        self.assertEqual(flat["hypotheses"]["4h|zx|short"]["evidence"], "no_edge")

    def test_report_says_how_much_evidence_the_observed_effect_would_need(self):
        doc = research.register(self.spans(), now=NOW)
        res = research.evaluate(self.family_events({"4h|zx|long": 0.45}, n=6000), doc)
        need = res["hypotheses"]["4h|zx|long"]["n_eff_needed_for_observed_effect"]
        self.assertIsNotNone(need)
        self.assertGreater(need, 0)

    def test_only_the_judge_can_write_the_allow_list(self):
        research.register(self.spans(), now=NOW)
        with self.assertRaises(gates.GateError):
            gates.set_allowed_combos(["4h|zx|long"], "manual")


class TrustHysteresisTests(unittest.TestCase):
    def test_one_lucky_build_does_not_grant_trust(self):
        table = {"trust_history": {"4h": {"policy_model": [False, True]}}}
        self.assertFalse(calib.trusted_with_hysteresis("4h", "policy_model", table))

    def test_two_consecutive_good_builds_grant_trust(self):
        table = {"trust_history": {"4h": {"policy_model": [False, True, True]}}}
        self.assertTrue(calib.trusted_with_hysteresis("4h", "policy_model", table))

    def test_a_single_failure_revokes_trust_immediately(self):
        table = {"trust_history": {"4h": {"policy_model": [True, True, False]}}}
        self.assertFalse(calib.trusted_with_hysteresis("4h", "policy_model", table))

    def test_alternating_results_never_grant_trust(self):
        runs = []
        for k in range(8):
            runs.append(k % 2 == 0)
            table = {"trust_history": {"1h": {"model": list(runs)}}}
            self.assertFalse(calib.trusted_with_hysteresis("1h", "model", table))

    def test_version_change_resets_the_history(self):
        old = {"version": 1, "trust_history": {"4h": {"model": [True, True]}}}
        new = {"version": 2, "tfs": {"4h": {}}}
        hist = calib._update_trust_history(old, new)
        self.assertEqual(hist["4h"]["model"], [False])

    def test_rebuild_is_weekly_not_daily(self):
        self.assertEqual(calib.REBUILD_SEC, 7 * 86400)


class DegeneratePolicyTests(unittest.TestCase):
    def test_zero_slope_policy_with_edge_weight_is_never_trusted(self):
        pm = {"edge_weight": 1.0, "edge_calibration": [0.0, -0.11],
              "live_norm": {"edge_sd": 1e-6, "p_sd": 0.02},
              "test": {"n": 999, "avg_net_r": 0.5, "profit_factor": 3, "uplift_r": 0.5,
                       "lcb_net_r": 0.3},
              "test_halves_avg_r": [0.4, 0.4]}
        self.assertTrue(calib._policy_is_degenerate(pm))
        self.assertFalse(calib._policy_model_trusted(pm))

    def test_healthy_policy_is_not_flagged(self):
        pm = {"edge_weight": 0.5, "edge_calibration": [0.54, 0.11],
              "live_norm": {"edge_sd": 0.1, "p_sd": 0.03}}
        self.assertFalse(calib._policy_is_degenerate(pm))


if __name__ == "__main__":
    unittest.main()
