# -*- coding: utf-8 -*-
"""تست‌های اصلاحِ لایهٔ calib و ورودی‌های زندهٔ مدل (ممیزیِ ۲۰۲۶-۰۹-۲۴، گروهِ calib).

هر کلاس یک یافته را می‌پاید: اگر آموزش و اجرا دو تعریفِ متفاوت از یک ویژگی بسازند،
یا آماری از پنجرهٔ آزمون نشت کند، این‌جا قرمز می‌شود.
"""
import os
import sys
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


if __name__ == "__main__":
    unittest.main()
