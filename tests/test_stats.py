# -*- coding: utf-8 -*-
"""تست‌های پذیرشِ موتورِ آماری (فاز ۲).

معیارِ اصلی: روی دادهٔ **بدونِ لبه**، نرخِ عبورِ کاذب باید نزدیکِ alpha باشد،
نه ۸۳ تا ۹۹ درصدی که معیارهای قبلی می‌دادند.
"""
import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import stats  # noqa: E402

HOUR = 3_600_000
BLOCK = 40 * HOUR                      # طولِ بلوک = افقِ برچسب (۴۰ کندلِ یک‌ساعته)


def r_draws(rng, n, edge=0.0):
    """توزیعی شبیهِ ساختارِ 1.8R/−1R با میانگینِ کنترل‌شده."""
    u = rng.random(n)
    base = np.where(u < 0.643, -1.0, 1.8)          # میانگینِ ~۰ در نرخِ بردِ سربه‌سر
    return base - base.mean() + edge


class EffectiveNTests(unittest.TestCase):
    def test_overlapping_labels_collapse_toward_one_observation(self):
        ts = np.repeat(np.arange(10) * BLOCK, 12)   # ۱۲ نمونهٔ هم‌پوشان در هر بلوک
        vals = np.zeros(len(ts))
        n_eff = stats.effective_n(vals, ts, BLOCK)
        self.assertEqual(len(vals), 120)
        self.assertLess(n_eff, 30, "n مؤثر باید بسیار کمتر از n اسمی باشد")
        self.assertGreater(n_eff, 9)

    def test_independent_events_keep_their_sample_size(self):
        ts = np.arange(50) * BLOCK * 3              # هر رویداد در بلوکِ خودش
        n_eff = stats.effective_n(np.zeros(50), ts, BLOCK)
        self.assertAlmostEqual(n_eff, 50.0)

    def test_higher_cross_sectional_correlation_shrinks_n_more(self):
        ts = np.repeat(np.arange(5) * BLOCK, 20)
        low = stats.effective_n(np.zeros(100), ts, BLOCK, rho=0.1)
        high = stats.effective_n(np.zeros(100), ts, BLOCK, rho=0.9)
        self.assertGreater(low, high)


class BootstrapTests(unittest.TestCase):
    def test_lower_bound_is_below_the_mean_and_reproducible(self):
        rng = np.random.default_rng(1)
        vals = r_draws(rng, 400, edge=0.2)
        ts = np.arange(400) * HOUR
        a = stats.block_bootstrap_lcb(vals, ts, BLOCK, B=500)
        b = stats.block_bootstrap_lcb(vals, ts, BLOCK, B=500)
        self.assertEqual(a, b, "قضاوتِ یک‌بارمصرف باید تکرارپذیر باشد")
        self.assertLess(a, float(vals.mean()))

    def test_real_edge_is_detected_with_enough_independent_events(self):
        rng = np.random.default_rng(7)
        vals = r_draws(rng, 3000, edge=0.25)
        ts = np.arange(3000) * BLOCK          # مستقل
        self.assertGreater(stats.block_bootstrap_lcb(vals, ts, BLOCK, B=400), 0)

    def test_no_edge_is_not_detected(self):
        rng = np.random.default_rng(11)
        vals = r_draws(rng, 3000, edge=0.0)
        ts = np.arange(3000) * BLOCK
        self.assertLessEqual(stats.block_bootstrap_lcb(vals, ts, BLOCK, B=400), 0)

    def test_single_cluster_returns_unknown_not_false_confidence(self):
        """همان حالتِ «جیبِ زنده»: n=۱۸ با میانگینِ به‌ظاهر عالی، همه در یک بازه."""
        vals = np.array([1.8] * 8 + [-1.0] * 10, dtype=float)   # میانگین ~ +۰٫۲۴R
        ts = np.arange(18) * HOUR                                # همه در یک بلوک
        self.assertGreater(float(vals.mean()), 0.2)
        self.assertIsNone(stats.block_bootstrap_lcb(vals, ts, BLOCK, B=400),
                          "با یک خوشه نباید کرانِ پایینِ مطمئن ساخت")

    def test_tiny_bucket_spread_over_time_still_fails_the_bar(self):
        vals = np.array([1.8] * 8 + [-1.0] * 10, dtype=float)
        ts = np.arange(18) * BLOCK * 2                           # ۱۸ بلوکِ جدا
        self.assertGreater(float(vals.mean()), 0.2)
        self.assertLessEqual(stats.block_bootstrap_lcb(vals, ts, BLOCK, B=400), 0)


class FalseDiscoveryRateTests(unittest.TestCase):
    def test_zero_edge_buckets_rarely_pass_the_lower_bound(self):
        """معیارِ قدیمیِ edge_book روی ۱۲ سطل ۸۳٪ کشفِ کاذب می‌داد."""
        rng = np.random.default_rng(3)
        hits = 0
        trials = 120
        for _ in range(trials):
            passed = False
            for _bucket in range(12):
                vals = r_draws(rng, 18)
                ts = np.sort(rng.integers(0, 30 * 24, 18)) * HOUR
                lcb = stats.block_bootstrap_lcb(vals, ts, BLOCK, B=200,
                                                seed=int(rng.integers(1, 10 ** 6)))
                if lcb is not None and lcb > 0:     # None = نامعلوم = عبور نمی‌کند
                    passed = True
                    break
            hits += passed
        rate = hits / trials
        self.assertLess(rate, 0.25, f"نرخِ کشفِ کاذب {rate:.0%} — هنوز بالاست")


class RomanoWolfTests(unittest.TestCase):
    def test_family_of_pure_noise_is_not_rejected(self):
        rng = np.random.default_rng(23)
        fam = {}
        for k in range(8):
            ts = np.arange(300) * BLOCK
            fam[f"h{k}"] = (r_draws(rng, 300), ts)
        res = stats.romano_wolf(fam, BLOCK, B=300)
        self.assertFalse(any(v["reject_null"] for v in res.values()),
                         f"نویز نباید رد شود: {res}")

    def test_a_genuine_edge_survives_the_family_correction(self):
        rng = np.random.default_rng(29)
        fam = {}
        for k in range(7):
            fam[f"noise{k}"] = (r_draws(rng, 600), np.arange(600) * BLOCK)
        fam["real"] = (r_draws(rng, 600, edge=0.45), np.arange(600) * BLOCK)
        res = stats.romano_wolf(fam, BLOCK, B=400)
        self.assertTrue(res["real"]["reject_null"], res)

    def test_adjusted_p_is_monotone_in_the_stepdown_order(self):
        rng = np.random.default_rng(31)
        fam = {f"h{k}": (r_draws(rng, 200, edge=0.05 * k), np.arange(200) * BLOCK)
               for k in range(5)}
        res = stats.romano_wolf(fam, BLOCK, B=200)
        ordered = sorted(res.items(), key=lambda kv: -kv[1]["t_stat"])
        ps = [v["p_adj"] for _k, v in ordered]
        self.assertEqual(ps, sorted(ps), "p تصحیح‌شده باید در ترتیبِ گام‌به‌گام نزولی نباشد")


class PowerTests(unittest.TestCase):
    def test_detecting_a_small_edge_needs_roughly_a_thousand_events(self):
        n = stats.power_sample_size(0.10, sd=1.25)
        self.assertGreater(n, 700)
        self.assertLess(n, 1400)

    def test_a_bigger_edge_needs_fewer_events(self):
        self.assertLess(stats.power_sample_size(0.30), stats.power_sample_size(0.10))

    def test_deflated_threshold_rises_with_the_number_of_trials(self):
        one = stats.deflated_mean_threshold(1, sd=1.25, n_eff=200)
        many = stats.deflated_mean_threshold(100, sd=1.25, n_eff=200)
        self.assertGreater(many, one)


if __name__ == "__main__":
    unittest.main()
