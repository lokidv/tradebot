# -*- coding: utf-8 -*-
"""دورِ دومِ ۱۵دقیقه‌ای: هم‌ترازکردنِ دادهٔ موقعیت‌گیری و مدلِ walk-forward هرگز از آینده نمی‌خوانند."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import micro  # noqa: E402
import micro2  # noqa: E402

T0 = 1_640_995_200_000       # 2022-01-01


def _k(n, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    w = np.abs(rng.normal(0, 0.001, n)) * c
    v = rng.lognormal(3, 0.6, n)
    return {"t": T0 + np.arange(n, dtype=np.int64) * micro.BAR_MS, "o": o, "h": np.maximum(o, c) + w,
            "l": np.minimum(o, c) - w, "c": c, "v": v, "tbv": v * rng.uniform(0.3, 0.7, n)}


def _extra(n, seed=2):
    rng = np.random.default_rng(seed)
    return {"oi": 1e5 * np.exp(np.cumsum(rng.normal(0, 0.003, n))), "acct": np.exp(rng.normal(0.3, 0.2, n)),
            "top": np.exp(rng.normal(0.2, 0.2, n)), "taker": np.exp(rng.normal(0, 0.3, n)),
            "prem": rng.normal(0, 3e-4, n)}


def _perturb(k, x, btc, d):
    k2 = {key: v.copy() for key, v in k.items()}
    for col in ("o", "h", "l", "c"):
        k2[col][d + 1:] *= 1.3
    k2["v"][d + 1:] *= 9
    k2["tbv"][d + 1:] = k2["v"][d + 1:] * 0.95
    x2 = {key: v.copy() for key, v in x.items()}
    for key in x2:
        x2[key][d + 1:] *= 3
    b2 = {"c": btc["c"].copy()}
    b2["c"][d + 1:] *= 0.5
    return k2, x2, b2


class AlignTests(unittest.TestCase):
    def test_metrics_row_must_be_one_step_older_than_the_bar_close(self):
        t_src = np.array([0, 300_000, 600_000, 900_000], dtype=np.int64)       # ردیف‌های ۵دقیقه‌ای
        v = np.array([1.0, 2.0, 3.0, 4.0])
        bar_t = np.array([0], dtype=np.int64)                                  # کندلِ 00:00 در 00:15 بسته می‌شود
        got = micro2.align(t_src, v, bar_t + micro2.BAR_MS - micro2.METRIC_LAG_MS)
        self.assertEqual(got[0], 3.0)                                          # ردیفِ 00:10، نه 00:15

    def test_stale_rows_are_missing(self):
        got = micro2.align(np.array([0], dtype=np.int64), np.array([1.0]), np.array([2 * 3_600_000]))
        self.assertTrue(np.isnan(got[0]))


class NoLookAheadTests(unittest.TestCase):
    def test_rule_signals_ignore_the_future(self):
        n, d = 4000, 3500
        k, x, btc = _k(n), _extra(n), {"c": _k(n, seed=9)["c"]}
        f = micro2.features(k, x, btc)
        k2, x2, b2 = _perturb(k, x, btc, d)
        f2 = micro2.features(k2, x2, b2)
        for h in micro2.HYPOTHESES[:-1]:
            self.assertTrue(np.array_equal(micro2.signals(h, k, f)[:d + 1], micro2.signals(h, k2, f2)[:d + 1]), h)

    def test_walk_forward_model_ignores_the_future(self):
        n = 22_500                                               # تا اواخرِ اوت ۲۰۲۲ ⇒ مدل‌های ژوئیه و اوت
        d = n - 500
        k, x, btc = _k(n), _extra(n), {"c": _k(n, seed=9)["c"]}
        s1 = micro2.model_signals(k, micro2.features(k, x, btc))
        self.assertGreater(np.count_nonzero(s1), 50)
        k2, x2, b2 = _perturb(k, x, btc, d)
        s2 = micro2.model_signals(k2, micro2.features(k2, x2, b2))
        self.assertTrue(np.array_equal(s1[:d + 1], s2[:d + 1]))


class ModelTests(unittest.TestCase):
    def test_logistic_recovers_the_sign_of_a_real_effect(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20_000, 3))
        y = (rng.uniform(size=20_000) < 1 / (1 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 2])))).astype(float)
        w = micro2.fit_logistic(X, y)
        self.assertGreater(w[1], 0.6)
        self.assertLess(w[3], -0.35)
        self.assertLess(abs(w[2]), 0.1)


if __name__ == "__main__":
    unittest.main()
