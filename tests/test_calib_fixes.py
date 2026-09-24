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


if __name__ == "__main__":
    unittest.main()
