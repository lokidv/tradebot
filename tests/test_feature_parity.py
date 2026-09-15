# -*- coding: utf-8 -*-
"""تست‌های پذیرشِ فاز ۱e — یک بردارِ ویژگی، دو مسیر.

اگر مسیرِ آموزش و مسیرِ اجرا برای یک کندل دو بردارِ متفاوت بسازند، مدل روی
چیزی آموزش دیده که هرگز در زمانِ تصمیم نمی‌بیند.
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
import engine  # noqa: E402
import features  # noqa: E402

from test_bracket_contract import synthetic_klines  # noqa: E402


class FeatureSetTests(unittest.TestCase):
    def test_dead_dxy_feature_is_gone(self):
        self.assertNotIn("dxy_dir", engine.FEATS)
        self.assertEqual(len(engine.FEATS), 23)
        self.assertEqual(features.N_FEATS, 23)

    def test_feature_vector_length_matches_the_declared_names(self):
        kl = synthetic_klines()
        cs = engine.component_series(*[np.array(kl[k], float) for k in "ohlcv"])
        vec = features.build(cs, np.array(kl["c"], float), 300, 1, "zx",
                             kl["t"][300], features.context())
        self.assertEqual(len(vec), len(engine.FEATS))

    def test_model_version_was_bumped_with_the_feature_change(self):
        self.assertGreaterEqual(calib.CALIB_VERSION, 19)


class TrainServeParityTests(unittest.TestCase):
    """همان کندل، یک‌بار از مسیرِ آموزش و یک‌بار از مسیرِ اجرا."""

    def test_identical_context_yields_identical_vectors(self):
        kl = synthetic_klines(seed=17)
        c = np.array(kl["c"], float)
        cs = engine.component_series(*[np.array(kl[k], float) for k in "ohlcv"])
        i = 350
        ctx_kwargs = dict(funding_z=1.7, rs_rank=0.82, htf_sign=1, btc_align=True,
                          gold_m=0.3, breadth_m=-0.4, dom_m=0.15, ethbtc_m=-0.2)

        training = engine.event_features(cs, c, i, 1, "zx", kl["t"][i], **ctx_kwargs)
        live_extras = {"funding_z": 1.7, "rs_rank": 0.82, "htf_sign": 1,
                       "gold": 0.3, "breadth": -0.4, "dom": 0.15, "ethbtc": -0.2}
        serving = features.build(cs, c, i, 1, "zx", kl["t"][i],
                                 features.context_from_extras(live_extras, btc_align=True))
        np.testing.assert_allclose(training, serving, atol=1e-12)

    def test_extras_mapping_covers_every_context_key(self):
        ctx = features.context_from_extras(
            {"funding_z": 1.0, "rs_rank": 0.9, "htf_sign": -1,
             "gold": 0.1, "breadth": 0.2, "dom": 0.3, "ethbtc": 0.4}, btc_align=False)
        self.assertEqual(
            set(ctx), {"funding_z", "rs_rank", "htf_sign", "btc_align",
                       "gold_m", "breadth_m", "dom_m", "ethbtc_m"})
        self.assertEqual(ctx["breadth_m"], 0.2)
        self.assertFalse(ctx["btc_align"])

    def test_missing_extras_fall_back_to_neutral_not_to_garbage(self):
        ctx = features.context_from_extras({})
        self.assertEqual(ctx["funding_z"], 0.0)
        self.assertEqual(ctx["rs_rank"], 0.5)       # وسطِ رتبه، نه ۰
        self.assertEqual(ctx["htf_sign"], 0)

    def test_live_funding_uses_the_training_formula(self):
        rows = [[1, -0.4], [2, 0.9], [3, 2.31]]
        with mock.patch.object(features.market, "funding_z_map", return_value=rows):
            self.assertAlmostEqual(features.live_funding_z("BTCUSDT"), 2.31)
        with mock.patch.object(features.market, "funding_z_map", return_value=[]):
            self.assertEqual(features.live_funding_z("NEWUSDT"), 0.0)
        with mock.patch.object(features.market, "funding_z_map", side_effect=RuntimeError):
            self.assertEqual(features.live_funding_z("DEADUSDT"), 0.0)


class FeatureHealthTests(unittest.TestCase):
    def test_constant_feature_is_reported_dead(self):
        rows = [[1.0, 0.0, i * 0.1] for i in range(50)]
        rep = features.health(rows, names=["a", "always_zero", "c"])
        self.assertIn("always_zero", rep["dead"])
        self.assertFalse(rep["ok"])

    def test_sparse_but_varying_feature_is_only_low_coverage(self):
        rows = [[float(i), (0.0 if i % 5 else float(i)), -float(i)] for i in range(50)]
        rep = features.health(rows, names=["a", "sparse", "c"])
        self.assertIn("sparse", rep["low_coverage"])
        self.assertNotIn("sparse", rep["dead"])
        self.assertTrue(rep["ok"])       # کم‌پوشش کشنده نیست، فقط ضعیف

    def test_real_training_matrix_has_no_dead_feature_after_dxy_removal(self):
        kl = synthetic_klines(n=2000, seed=5)
        events, _z, _dx, _dy, _dr = calib.extract_events(
            "TESTUSDT", kl, "1h", [], {}, None, None)
        self.assertGreater(len(events), 20)
        rep = features.health([e["feats"] for e in events])
        self.assertEqual(rep["n_feat"], len(engine.FEATS))
        # روی دادهٔ مصنوعی، ویژگی‌های بازار-محور ثابتِ صفرند؛ فقط ویژگی‌های
        # قیمتی باید واریانس داشته باشند
        for name in ("z_dir", "votes", "trend_w", "atr_pct", "rsi_dir", "kdist_dir"):
            self.assertNotIn(name, rep["dead"], f"{name} نباید بی‌واریانس باشد")

    def test_psi_flags_a_shifted_distribution(self):
        train = [float(i % 10) for i in range(500)]
        same = [float(i % 10) for i in range(500)]
        shifted = [float(i % 10) + 6 for i in range(500)]
        self.assertLess(features.psi(train, same), 0.01)
        self.assertGreater(features.psi(train, shifted), features.PSI_WARN)


if __name__ == "__main__":
    unittest.main()
