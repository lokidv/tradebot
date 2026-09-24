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

if __name__ == "__main__":
    unittest.main()
