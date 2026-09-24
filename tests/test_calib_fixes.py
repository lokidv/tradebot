# -*- coding: utf-8 -*-
"""تست‌های اصلاحِ لایهٔ calib و ورودی‌های زندهٔ مدل (ممیزیِ ۲۰۲۶-۰۹-۲۴، گروهِ calib).

هر کلاس یک یافته را می‌پاید: اگر آموزش و اجرا دو تعریفِ متفاوت از یک ویژگی بسازند،
یا آماری از پنجرهٔ آزمون نشت کند، این‌جا قرمز می‌شود.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import calib  # noqa: E402


class TiedPredictionLiftTests(unittest.TestCase):
    """calib-F11: پیش‌بینیِ ثابت نباید روندِ زمانیِ پنجرهٔ آزمون را «لیفت» گزارش کند."""

    def test_constant_prediction_has_zero_lift(self):
        const = np.full(300, 0.2)
        trend = np.arange(300, dtype=float)                 # برچسبی که فقط با زمان بالا می‌رود
        self.assertEqual(calib._edge_lift(const, trend), 0.0)
        self.assertEqual(calib._lift_on(const, (trend > 150).astype(float)), 0.0)
        lift, _mask = calib._lift(np.full(300, 0.4), (trend > 150).astype(float))
        self.assertEqual(lift, 0.0)
        self.assertEqual(calib._rank_corr(const, trend), 0.0)

    def test_ties_are_not_broken_by_time_order(self):
        # ۵۴۰ پیش‌بینیِ برابر که برچسبشان فقط زمان است + ۶۰ پیش‌بینیِ بالاتر با برچسبِ میانه.
        # شکستنِ تساوی به ترتیبِ زمان لیفتِ ~۳۲۰ می‌ساخت؛ تصادفیِ ثابت باید نزدیکِ صفر بماند.
        pred = np.concatenate([np.zeros(540), np.ones(60)])
        actual = np.concatenate([np.arange(540, dtype=float), np.full(60, 270.0)])
        self.assertLess(abs(calib._edge_lift(pred, actual)), 80.0)
        # رتبهٔ میانگین: همهٔ مقدارهای برابر یک رتبه
        r = calib._avg_rank(np.array([3.0, 1.0, 3.0, 2.0]))
        np.testing.assert_allclose(r, [2.5, 0.0, 2.5, 1.0])

    def test_untied_predictions_keep_the_old_argsort_result(self):
        rng = np.random.RandomState(11)
        pred, actual = rng.normal(size=500), rng.normal(size=500)
        order = np.argsort(pred)
        third = len(order) // 3
        old = float(actual[order[-third:]].mean() - actual[order[:third]].mean())
        self.assertAlmostEqual(calib._edge_lift(pred, actual), old, places=12)
        ra = np.argsort(np.argsort(pred)).astype(float)
        rb = np.argsort(np.argsort(actual)).astype(float)
        self.assertAlmostEqual(calib._rank_corr(pred, actual), float(np.corrcoef(ra, rb)[0, 1]), places=12)


def _setup_model_blob(n_feat=3):
    return {"kind": "logit", "w": [0.1] + [0.3] * n_feat, "mu": [0.0] * n_feat, "sd": [1.0] * n_feat,
            "n_feat": n_feat, "n_train": 500, "n_oos": 200, "oos_lift": 12.0, "oos_brier": 0.20,
            "oos_base": 50.0, "platt": [1.0, 0.0], "avg_win_r": 1.8, "avg_loss_r": 1.0}


def _edge_model_blob(n_feat=3):
    return {"kind": "ridge_edge", "w": [0.05] + [0.1] * n_feat, "mu": [0.0] * n_feat,
            "sd": [1.0] * n_feat, "n_feat": n_feat, "calibration": [1.0, 0.0],
            "n_oos": 200, "oos_rank_ic": 0.10, "oos_lift_r": 0.30, "oos_mae_skill": 0.05,
            "residual_sd": 0.8, "trained_cost_pct": 0.15}


class SetupAndEdgeHysteresisTests(unittest.TestCase):
    """calib-F12: مدلِ ستاپ و مدلِ edge هم مثلِ بقیه فقط پس از دو ساختِ پیاپیِ موفق."""

    def _table(self, runs):
        return {"version": calib.CALIB_VERSION, "built_at": 0.0,
                "tfs": {"4h": {"model": _setup_model_blob(), "edge_model": _edge_model_blob(),
                               "cells": {}}},
                "trust_history": {"4h": {"model": list(runs), "edge_model": list(runs)}}}

    def test_one_lucky_build_gives_no_setup_probability_or_edge(self):
        with mock.patch.object(calib, "load", return_value=self._table([False, True])):
            out = calib.predict("4h", [0.5, 0.2, -0.1], 1.0, legacy=None, cost=0.1)
            st = calib.status()
        self.assertTrue(out is None or out.get("source") != "model")
        m = st["models"]["4h"]
        self.assertTrue(m["trusted_this_build"])
        self.assertFalse(m["trusted"])                        # tf_trust.setup_ok همان predict است
        self.assertTrue(m["edge_trusted_this_build"])
        self.assertFalse(m["edge_trusted"])

    def test_two_consecutive_good_builds_turn_both_on(self):
        with mock.patch.object(calib, "load", return_value=self._table([True, True])):
            out = calib.predict("4h", [0.5, 0.2, -0.1], 1.0, legacy=None, cost=0.1)
            st = calib.status()
        self.assertEqual(out["source"], "model")
        self.assertTrue(out["edge_trusted"])
        self.assertIsNotNone(out.get("edge_r"))
        self.assertTrue(st["models"]["4h"]["trusted"])
        self.assertTrue(st["models"]["4h"]["edge_trusted"])

    def test_edge_needs_its_own_history(self):
        table = self._table([True, True])
        table["trust_history"]["4h"]["edge_model"] = [True]   # اولین ساختِ نسخهٔ تازه
        with mock.patch.object(calib, "load", return_value=table):
            out = calib.predict("4h", [0.5, 0.2, -0.1], 1.0, legacy=None, cost=0.1)
        self.assertEqual(out["source"], "model")
        self.assertFalse(out["edge_trusted"])
        self.assertIsNone(out.get("edge_r"))


class _Stop(Exception):
    pass


class ActionPolicyEmbargoTests(unittest.TestCase):
    """calib-F8: برچسبِ سیاستِ عمل براکتِ ۴۰ کندلی است ⇒ embargo دست‌کم ۴۱ کندل."""

    def _dense(self, n=2000):
        rng = np.random.RandomState(5)
        dx, dy, dr = [], [], []
        for i in range(n):
            x = rng.normal(size=4).tolist()
            dx.append((x, [-v for v in x]))
            dy.append(((i // 5) * calib.TF_MS["4h"], 1.0))
            dr.append((0.1, -0.1, 1.0, 0.1))
        return dx, dy, dr

    def test_walk_forward_partitions_and_bootstrap_use_the_bracket_length(self):
        seen = {}

        def wf_edge(X, y, ts, embargo_ms):
            seen["wf"] = embargo_ms
            raise _Stop()
        dx, dy, dr = self._dense()
        with mock.patch.object(calib, "_walk_forward_edge", side_effect=wf_edge):
            with self.assertRaises(_Stop):
                calib._fit_action_policy(dx, dy, dr, "4h", calib.COST_PCT)
        self.assertEqual(seen["wf"], (calib.bracket.MAX_BARS + 1) * calib.TF_MS["4h"])
        self.assertGreater(seen["wf"], calib.engine.HORIZON["4h"] * calib.TF_MS["4h"])

    def test_partition_purge_matches(self):
        seen = {}
        real_parts = calib._time_partitions

        def parts(rows, timestamps, purge_ms=0):
            seen["purge"] = purge_ms
            return real_parts(rows, timestamps, purge_ms=purge_ms)
        dx, dy, dr = self._dense()
        nan_oof = lambda X, y, ts, emb, *a: {"ridge_edge": np.zeros(len(y))}   # noqa: E731
        with mock.patch.object(calib, "_walk_forward_edge", side_effect=nan_oof), \
                mock.patch.object(calib, "_walk_forward_ridge_rolling",
                                  side_effect=lambda X, y, ts, emb, w: np.zeros(len(y))), \
                mock.patch.object(calib, "_time_partitions", side_effect=parts):
            calib._fit_action_policy(dx, dy, dr, "1h", calib.COST_PCT)
        self.assertEqual(seen["purge"], (calib.bracket.MAX_BARS + 1) * calib.TF_MS["1h"])


def _klines(n, seed, start_bar=0, bar_ms=3_600_000):
    from test_bracket_contract import synthetic_klines
    kl = synthetic_klines(n=n, seed=seed)
    kl["t"] = [(start_bar + k) * bar_ms for k in range(n)]
    return kl


class DenseGridTests(unittest.TestCase):
    """calib-F13/F14: نمونه‌های متراکم روی شبکهٔ زمانیِ سراسری و فقط با براکتِ کامل."""

    def test_coins_listed_at_different_offsets_share_the_same_timestamps(self):
        stamps = []
        for k, offset in enumerate((0, 7, 14, 21, 29)):
            _e, _z, _dx, dy, _dr = calib.extract_events(
                f"C{k}USDT", _klines(1200 - offset, seed=3 + k, start_bar=offset), "1h",
                [], {}, None, None)
            ts = {t for t, _y in dy}
            self.assertTrue(all((t // calib.TF_MS["1h"]) % calib.DENSE_STRIDE == 0 for t in ts))
            stamps.append(ts)
        common = set.intersection(*stamps)
        self.assertGreater(len(common), 150)          # قبلاً ۰: هر ارز در ردهٔ باقی‌ماندهٔ خودش

    def test_every_dense_bracket_has_the_full_forty_bars(self):
        n = 1000
        kl = _klines(n, seed=9)
        _e, _z, _dx, dy, _dr = calib.extract_events("AUSDT", kl, "1h", [], {}, None, None)
        idx = {t: i for i, t in enumerate(kl["t"])}
        last_i = max(idx[t] for t, _y in dy)
        self.assertLessEqual(last_i + calib.bracket.MAX_BARS, n - 1)
        self.assertLessEqual(last_i + calib.engine.HORIZON["1h"], n - 1)

    def test_model_version_was_bumped(self):
        self.assertGreaterEqual(calib.CALIB_VERSION, 23)


class LiveCostAdjustmentTests(unittest.TestCase):
    """calib-F2: تعدیلِ زنده = هزینهٔ همین معامله در R − میانگینِ هزینهٔ R که مدل یاد گرفته."""

    def test_delta_uses_live_cost_over_live_risk_minus_the_trained_mean(self):
        m = {"trained_cost_r": 0.16, "live_funding_pct": 0.025}
        got = calib._live_cost_delta_r(m, 0.08, 0.634)
        self.assertAlmostEqual(got, (0.08 + 0.025) / 0.634 - 0.16, places=12)
        # معامله‌ای با هزینه/ریسکِ برابرِ میانگینِ آموزش هیچ تعدیلی نمی‌گیرد
        self.assertAlmostEqual(calib._live_cost_delta_r(m, 0.16 - 0.025, 1.0), 0.0, places=12)

    def test_old_tables_keep_the_old_formula(self):
        m = {"trained_cost_pct": 0.15}
        self.assertAlmostEqual(calib._live_cost_delta_r(m, 0.30, 1.5), 0.10, places=12)

    def test_edge_model_stores_the_pooled_cost_in_r(self):
        rng = np.random.RandomState(17)
        events = []
        for i in range(1500):
            x = rng.normal(size=6)
            risk = 0.5 if i % 2 else 2.0
            events.append({"ts": i * calib.TF_MS["1h"], "feats": x.tolist(), "risk_pct": risk,
                           "r": 0.6 * x[0] + 0.1 * rng.normal(), "cost_pct": 0.10, "regime": "trend"})
        m = calib._fit_edge_model(events, "1h", calib.COST_PCT)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m["trained_cost_r"], (0.10 / 0.5 + 0.10 / 2.0) / 2, delta=0.005)
        self.assertAlmostEqual(m["trained_cost_pct"], 0.10, places=6)
        self.assertEqual(m["live_funding_pct"],
                         calib.costs.expected_funding_pct("1h", calib.bracket.MAX_BARS / 2))
        # زنده: ریسکِ کم (۰٫۵٪) گران‌تر از میانگین است، ریسکِ زیاد (۲٪) ارزان‌تر
        feats = [0.0] * 6
        lo = calib._score_edge(m, feats, 0.5, 0.10)["edge_r"]
        hi = calib._score_edge(m, feats, 2.0, 0.10)["edge_r"]
        self.assertAlmostEqual(hi - lo, (0.10 + m["live_funding_pct"]) * (1 / 0.5 - 1 / 2.0), delta=0.002)

    def test_action_policy_stores_the_cost_meta(self):
        rng = np.random.RandomState(37)
        dx, dy, dr = [], [], []
        for i in range(8_000):
            x = rng.normal(size=6)
            long_x, short_x = x.copy(), x.copy()
            short_x[0] *= -1
            dx.append((long_x.tolist(), short_x.tolist()))
            dy.append(((i // 10) * calib.TF_MS["1h"], 1.0))
            dr.append((float(0.72 * long_x[0] + 0.15), float(0.72 * short_x[0] + 0.15), 1.0, 0.2))
        with mock.patch.object(calib, "HAS_LGBM", False):
            m = calib._fit_action_policy(dx, dy, dr, "1h", calib.COST_PCT)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m["trained_cost_r"], 0.2, places=6)
        s1 = calib._score_action_policy(m, [2, 0, 0, 0, 0, 0], [-2, 0, 0, 0, 0, 0], 1.0, 0.2)
        s2 = calib._score_action_policy(m, [2, 0, 0, 0, 0, 0], [-2, 0, 0, 0, 0, 0], 1.0, 0.4)
        self.assertEqual(s1["side"], s2["side"])                      # هزینه سمت را عوض نمی‌کند
        self.assertAlmostEqual(s1["edge_r"] - s2["edge_r"], 0.2, delta=0.002)


T0_4H = 1_640_995_200_000          # 2022-01-01، مرزِ کندلِ 4h
BAR_4H = 14_400_000


class _BuildHarness:
    """``calib.build`` روی تاریخچهٔ مصنوعی، بی‌شبکه؛ آموزش‌دهنده‌ها فقط ورودی را ثبت می‌کنند."""

    SYMS = ("BTCUSDT", "ETHUSDT", "XUSDT")

    def __init__(self, n=1500, prereg=None, judged=False):
        self.n, self.prereg, self.judged = n, prereg, judged
        self.seen, self.judge_input = {}, None
        self.hists = {s: _klines(n, seed=21 + k, start_bar=T0_4H // BAR_4H, bar_ms=BAR_4H)
                      for k, s in enumerate(self.SYMS)}

    def _hist(self, sym, tf, bars=3000, **_k):
        if sym not in self.hists:
            raise RuntimeError("بی‌شبکه")
        return self.hists[sym]

    def _rec(self, name, ret=None):
        def f(*a, **_k):
            self.seen[name] = a
            return ret
        return f

    def _judge(self, events_by_tf):
        self.judge_input = events_by_tf
        return {"passing": []}

    def run(self, tfs=("4h",)):
        tmp = tempfile.TemporaryDirectory()
        pats = [
            mock.patch.object(calib, "CALIB_PATH", os.path.join(tmp.name, "calib.json")),
            mock.patch.object(calib.market, "get_history", side_effect=self._hist),
            mock.patch.object(calib.market, "funding_z_map", return_value=[]),
            mock.patch.object(calib.market, "get_funding_history", return_value=[]),
            mock.patch.object(calib.research, "ensure_registered", return_value=(self.prereg, False)),
            mock.patch.object(calib.research, "last_judgement",
                              return_value={"judged": self.judged}),
            mock.patch.object(calib.research, "judge", side_effect=self._judge),
            mock.patch.object(calib, "_train_logistic", side_effect=self._rec("model")),
            mock.patch.object(calib, "_fit_edge_model", side_effect=self._rec("edge")),
            mock.patch.object(calib, "_fit_policy_model", side_effect=self._rec("policy")),
            mock.patch.object(calib, "_train_dir_model", side_effect=self._rec("dir")),
            mock.patch.object(calib, "_fit_action_policy", side_effect=self._rec("action")),
            mock.patch.object(calib, "_table", None),
        ]
        for p in pats:
            p.start()
        try:
            calib.build(list(self.SYMS), tfs=tfs)
            st = dict(calib._state)
            table = calib._table
        finally:
            for p in reversed(pats):
                p.stop()
            tmp.cleanup()
        if st.get("error"):
            raise AssertionError(st["error"])
        return table


def _prereg(start_bar, end_bar):
    return {"hash": "x" * 64, "final_windows": {
        "4h": {"start_ms": T0_4H + start_bar * BAR_4H, "end_ms": T0_4H + end_bar * BAR_4H}}}


class FrozenWindowTests(unittest.TestCase):
    """calib-F9: سطل‌ها و purge؛ calib-F15: پس از داوری پنجره به آموزش برمی‌گردد."""

    def test_window_events_are_out_of_cells_and_labels_before_it_are_purged(self):
        doc = _prereg(900, 1200)
        h = _BuildHarness(prereg=doc, judged=False)
        table = h.run()
        w = doc["final_windows"]["4h"]
        trained = h.seen["model"][0]
        self.assertGreater(len(trained), 10)
        purge_lo = w["start_ms"] - (calib.bracket.MAX_BARS + 1) * BAR_4H
        self.assertFalse([e for e in trained if purge_lo <= e["ts"] < w["end_ms"]])
        dense_ts = [yy[0] for yy in h.seen["dir"][1]]
        self.assertFalse([t for t in dense_ts if purge_lo <= t < w["end_ms"]])
        blob = table["tfs"]["4h"]
        self.assertEqual(blob["cells"]["all"][0], len(trained))      # سطل = فقط آموزش
        held = h.judge_input["4h"]
        self.assertTrue(held)
        self.assertTrue(all(w["start_ms"] <= e["ts"] < w["end_ms"] for e in held))
        self.assertEqual(blob["held_out_final_events"], len(held))
        self.assertEqual(table["frozen_window"], {"excluded": True, "judged": False})

    def test_after_the_one_shot_judgement_the_window_is_trained_on(self):
        doc = _prereg(900, 1200)
        h = _BuildHarness(prereg=doc, judged=True)
        table = h.run()
        w = doc["final_windows"]["4h"]
        trained = h.seen["model"][0]
        self.assertTrue([e for e in trained if w["start_ms"] <= e["ts"] < w["end_ms"]])
        self.assertIsNone(h.judge_input)                             # داوری فقط یک‌بار
        self.assertEqual(table["tfs"]["4h"]["held_out_final_events"], 0)
        self.assertEqual(table["frozen_window"], {"excluded": False, "judged": True})
        self.assertEqual(table["tfs"]["4h"]["cells"]["all"][0], len(trained))


if __name__ == "__main__":
    unittest.main()
