# -*- coding: utf-8 -*-
"""تست‌های اصلاحِ لایهٔ calib و ورودی‌های زندهٔ مدل (ممیزیِ ۲۰۲۶-۰۹-۲۴، گروهِ calib).

هر کلاس یک یافته را می‌پاید: اگر آموزش و اجرا دو تعریفِ متفاوت از یک ویژگی بسازند،
یا آماری از پنجرهٔ آزمون نشت کند، این‌جا قرمز می‌شود.
"""
import json
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
import features  # noqa: E402
import market  # noqa: E402


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
            self.assertTrue(all(calib._dense_slot(t, calib.TF_MS["1h"]) for t in ts))
            stamps.append(ts)
        common = set.intersection(*stamps)
        self.assertGreater(len(common), 150)          # قبلاً ۰: هر ارز در ردهٔ باقی‌ماندهٔ خودش
        # چرخشِ روزانه: همهٔ ساعت‌های روز نمونه دارند (باقی‌ماندهٔ ثابت فقط ۶ ساعت از ۲۴ بود)
        self.assertEqual({(t // 3_600_000) % 24 for t in stamps[0]}, set(range(24)))
        self.assertAlmostEqual(len(stamps[0]) / (1200 - 40 - 260), 1 / calib.DENSE_STRIDE, delta=0.02)

    def test_grid_rotation_covers_every_4h_slot_and_keeps_daily_density(self):
        bar4, day = calib.TF_MS["4h"], 86_400_000
        picked = [k * bar4 for k in range(6 * 400) if calib._dense_slot(k * bar4, bar4)]
        self.assertEqual({(t % day) // bar4 for t in picked}, set(range(6)))
        self.assertAlmostEqual(len(picked) / (6 * 400), 0.25, delta=0.01)
        days = [d * day for d in range(400) if calib._dense_slot(d * day, day)]
        self.assertEqual({(t // day + 4) % 7 for t in days}, set(range(7)))   # همهٔ روزهای هفته
        self.assertAlmostEqual(len(days) / 400, 0.25, delta=0.01)

    def test_every_dense_bracket_has_the_full_forty_bars(self):
        n = 1000
        kl = _klines(n, seed=9)
        _e, _z, _dx, dy, _dr = calib.extract_events("AUSDT", kl, "1h", [], {}, None, None)
        idx = {t: i for i, t in enumerate(kl["t"])}
        last_i = max(idx[t] for t, _y in dy)
        self.assertLessEqual(last_i + calib.bracket.MAX_BARS, n - 1)
        self.assertLessEqual(last_i + calib.engine.HORIZON["1h"], n - 1)

    def test_thinning_keeps_whole_timestamps_across_the_whole_history(self):
        # گروه‌های نابرابر، مثلِ ارزهایی که دیرتر لیست شده‌اند
        ts = sorted(t for t in range(5000) for _ in range(3 + (t * 7) % 11))
        keep = calib._thin_whole_timestamps(ts, 10_000)
        self.assertLessEqual(len(keep), 10_000)
        self.assertGreater(len(keep), 9_000)
        kept = np.asarray(ts)[keep]
        full = dict(zip(*np.unique(ts, return_counts=True)))
        for t, c in zip(*np.unique(kept, return_counts=True)):
            self.assertEqual(c, full[t])                  # هیچ لحظه‌ای نیمه‌کاره نمی‌ماند
        self.assertEqual((kept[0], kept[-1]), (0, 4999))  # سراسرِ تاریخ
        self.assertEqual(list(calib._thin_whole_timestamps(ts[:50], 100)), list(range(50)))

    def test_action_policy_thinning_does_not_split_cross_sections(self):
        seen = {}

        def wf_edge(X, y, ts, embargo_ms):
            seen["ts"] = ts[0::2]
            raise _Stop()
        dx, dy, dr = ActionPolicyEmbargoTests()._dense(n=3000)
        with mock.patch.object(calib, "ACTION_MAX_SAMPLES", 1_700), \
                mock.patch.object(calib, "_walk_forward_edge", side_effect=wf_edge):
            with self.assertRaises(_Stop):
                calib._fit_action_policy(dx, dy, dr, "4h", calib.COST_PCT)
        self.assertLessEqual(len(seen["ts"]), 1_700)
        _u, counts = np.unique(seen["ts"], return_counts=True)
        self.assertTrue((counts == 5).all())              # هر لحظه با هر پنج ارزش

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

    def test_event_models_use_the_funding_their_events_actually_paid(self):
        rng = np.random.RandomState(19)
        events = []
        for i in range(1500):
            x = rng.normal(size=6)
            events.append({"ts": i * calib.TF_MS["4h"], "feats": x.tolist(), "risk_pct": 1.5,
                           "r": 0.6 * x[0] + 0.1 * rng.normal(), "cost_pct": 0.11 + 0.02,
                           "tier_pct": 0.11, "regime": "trend"})
        m = calib._fit_edge_model(events, "4h", calib.COST_PCT)
        self.assertAlmostEqual(m["live_funding_pct"], 0.02, places=6)
        # همان رده و همان ریسکِ آموزش ⇒ هیچ تعدیلی (قبلاً برآوردِ ثابتِ ۰٫۱٪ فاندینگ اضافه می‌شد)
        self.assertAlmostEqual(calib._live_cost_delta_r(m, 0.11, 1.5), 0.0, places=6)

    def test_extracted_events_carry_their_tier(self):
        kl = _klines(2000, seed=5)
        events, *_ = calib.extract_events("TESTUSDT", kl, "1h", [], {}, None, None)
        self.assertTrue(events)
        for e in events:
            self.assertGreaterEqual(e["cost_pct"], e["tier_pct"])
            self.assertIn(e["tier_pct"], [t for _f, t in calib.costs.TIERS])

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


H8 = 8 * 3_600_000


class FundingAlignmentTests(unittest.TestCase):
    """calib-F3: یک قاعده در آموزش و اجرا — آخرین تسویهٔ پیش از بسته‌شدنِ کندل."""

    def test_settlement_at_the_close_is_excluded_and_at_the_open_included(self):
        rows = [[0, 0.1], [H8 + 3, 0.2], [2 * H8 + 2, 0.3]]        # +ms مثلِ fundingTimeِ بایننس
        bar = 4 * 3_600_000
        # کندلِ ۰۴:۰۰-۰۸:۰۰: تسویهٔ ۰۸:۰۰ روی مرزِ بسته‌شدن است ⇒ هنوز ۰٫۱
        self.assertEqual(features.funding_z_at(rows, bar, bar), 0.1)
        # کندلِ ۰۸:۰۰-۱۲:۰۰: تسویهٔ ۰۸:۰۰ داخلِ کندل ⇒ ۰٫۲ (قاعدهٔ قدیمِ «≤ باز» ۰٫۱ می‌داد)
        self.assertEqual(features.funding_z_at(rows, 2 * bar, bar), 0.2)
        # روزانه: سه تسویهٔ روز، آخری (۱۶:۰۰) پیش از بسته‌شدن
        self.assertEqual(features.funding_z_at(rows, 0, 86_400_000), 0.3)
        self.assertEqual(features.funding_z_at([], 0, 86_400_000), 0.0)

    def test_live_value_is_the_one_training_sees_for_the_same_bar(self):
        kl = _klines(900, seed=4)
        fz = [[k * H8 + 1, float(np.sin(k / 3.0)) * 3] for k in range(0, 900 // 8 + 2)]
        _e, _z, dx, dy, _dr = calib.extract_events("AUSDT", kl, "1h", fz, {}, None, None)
        j = calib.engine.FEATS.index("funding_dir")
        checked = 0
        for (lf, _sf), (ts, _y) in zip(dx, dy):
            with mock.patch.object(features.market, "funding_z_map", return_value=fz):
                live = features.live_funding_z("AUSDT", ts, calib.TF_MS["1h"])
            self.assertAlmostEqual(lf[j], max(-1.0, min(1.0, live / 4.0)), places=12)
            checked += 1
        self.assertGreater(checked, 50)

    def test_live_without_a_bar_keeps_the_old_latest_row(self):
        with mock.patch.object(features.market, "funding_z_map", return_value=[[1, 0.5], [2, 1.5]]):
            self.assertEqual(features.live_funding_z("X"), 1.5)


class _FakeBinanceFunding:
    """/fapi/v1/fundingRate با startTime/endTime/limit — مثلِ بایننس، رو به جلو."""

    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def __call__(self, path, params=None):
        self.calls.append(dict(params))
        lo, hi = params.get("startTime", 0), params.get("endTime", 1 << 62)
        got = [r for r in self.rows if lo <= r[0] <= hi][: params.get("limit", 1000)]
        return [{"fundingTime": t, "fundingRate": v} for t, v in got]


class FundingHistoryPagingTests(unittest.TestCase):
    """calib-F4: تاریخچه تا ابتدای پنجرهٔ آموزش صفحه‌بندی می‌شود و کش هم‌منبع می‌ماند."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.now_ms = 1_790_000_000_000
        start = self.now_ms - 3000 * H8 - 3_600_000                     # آخرین تسویه یک ساعت پیش
        self.series = [[start + k * H8 + 2, float(1e-4 * np.sin(k))] for k in range(3001)]
        self.pats = [mock.patch.object(market, "HIST_DIR", self.tmp.name),
                     mock.patch.object(market, "FUNDING_PAGE_GAP", {}),
                     mock.patch.object(market.time, "time", lambda: self.now_ms / 1000.0)]
        for p in self.pats:
            p.start()
        market._neg_until.clear()

    def tearDown(self):
        for p in reversed(self.pats):
            p.stop()
        self.tmp.cleanup()

    def test_training_pages_back_and_live_reads_the_same_cache(self):
        since = self.now_ms - 2500 * H8
        fake = _FakeBinanceFunding(self.series)
        with mock.patch.object(market, "_fapi_json", side_effect=fake):
            rows = market.get_funding_history("BTCUSDT", since_ms=since)
            self.assertGreaterEqual(len(fake.calls), 3)                   # ۱۰۰۰ ردیف در هر صفحه
            self.assertLessEqual(rows[0][0], since - 30 * H8)             # گرم‌شدنِ z
            self.assertEqual(rows[-1], self.series[-1])
            n_calls = len(fake.calls)
            live = market.get_funding_history("BTCUSDT")                  # همان کش، بی شبکه
            self.assertEqual(len(fake.calls), n_calls)
        self.assertEqual(live, self.series[-1000:])

    def test_z_of_recent_settlements_is_the_same_for_training_and_live(self):
        since = self.now_ms - 2500 * H8
        with mock.patch.object(market, "_fapi_json", side_effect=_FakeBinanceFunding(self.series)):
            z_train = dict(market.funding_z_map("BTCUSDT", since_ms=since))
            z_live = dict(market.funding_z_map("BTCUSDT"))
        common = sorted(set(z_train) & set(z_live))[40:]
        self.assertGreater(len(common), 900)
        for t in common:
            self.assertAlmostEqual(z_train[t], z_live[t], places=12)

    def test_refresh_after_a_settlement_fetches_only_new_rows_from_the_same_source(self):
        fake = _FakeBinanceFunding(self.series[:-3])
        with mock.patch.object(market, "_fapi_json", side_effect=fake):
            market.get_funding_history("BTCUSDT")
        path = os.path.join(self.tmp.name, "funding_BTCUSDT.json")
        os.utime(path, (self.now_ms / 1000.0, self.now_ms / 1000.0))
        # سه تسویهٔ تازه منتشر شد؛ منبعِ کش بایننس است
        self.now_ms += 3 * H8 + 120_000
        fake2 = _FakeBinanceFunding(self.series)
        with mock.patch.object(market, "_fapi_json", side_effect=fake2), \
                mock.patch.object(market, "_get_json", side_effect=AssertionError("منبعِ دیگر")):
            rows = market.get_funding_history("BTCUSDT")
        self.assertEqual(len(fake2.calls), 1)
        self.assertGreater(fake2.calls[0]["startTime"], self.series[-5][0])
        self.assertEqual(rows[-1], self.series[-1])

    def test_cache_is_fresh_until_the_next_settlement_is_due(self):
        rows = [[0, 0.1], [H8, 0.2]]
        doc = {"src": "binance", "from_ms": 0, "rows": rows}
        mtime = 2 * H8 / 1000 - 3600                                     # یک ساعت پیش از موعد نوشته شد
        self.assertTrue(market._funding_doc_fresh(doc, mtime, now=(2 * H8 - 1000) / 1000))
        due = (2 * H8 + market.FUNDING_PUBLISH_GRACE_MS) / 1000
        self.assertFalse(market._funding_doc_fresh(doc, mtime, now=due + 1))
        self.assertTrue(market._funding_doc_fresh(doc, due + 1 - 10, now=due + 1))   # همین حالا چک شد
        self.assertFalse(market._funding_doc_fresh({"src": None, "rows": rows}, mtime, now=2.0))

    def test_blocked_binance_falls_back_to_deep_kucoin_history(self):
        def blocked(*a, **k):
            raise RuntimeError("451")
        pages = []

        def kucoin(url, params):
            pages.append(params)
            lo, hi = params["from"], params["to"]
            got = [r for r in self.series if lo <= r[0] <= hi][-100:][::-1]
            return {"code": "200000", "data": [{"timepoint": t, "fundingRate": v} for t, v in got]}
        since = self.now_ms - 2000 * H8
        with mock.patch.object(market, "_fapi_json", side_effect=blocked), \
                mock.patch.object(market, "_get_json", side_effect=kucoin):
            rows = market.get_funding_history("ETHUSDT", since_ms=since)
        self.assertGreater(len(pages), 15)
        self.assertLessEqual(rows[0][0], since)
        with open(os.path.join(self.tmp.name, "funding_ETHUSDT.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["src"], "kucoin")


class _FakeKucoinFunding:
    """/api/v1/contract/funding-rates با from/to — ۱۰۰ ردیفِ جدیدترِ بازه، جدید→قدیم."""

    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def __call__(self, url, params):
        self.calls.append(dict(params))
        lo, hi = params["from"], params["to"]
        got = [r for r in self.rows if lo <= r[0] <= hi][-100:][::-1]
        return {"code": "200000", "data": [{"timepoint": t, "fundingRate": v} for t, v in got]}


class FundingSourceFailoverTests(unittest.TestCase):
    """منبعِ کشِ فاندینگ که از کار افتاد و ردیف‌هایش کهنه شد به منبعِ بعدیِ زنجیره می‌رود
    (قبلاً سندِ هم‌منبع هرگز منبع عوض نمی‌کرد و فاندینگ بی‌صدا یخ می‌زد)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.now_ms = 1_790_000_000_000
        start = self.now_ms - 1500 * H8 - 3_600_000                     # آخرین تسویه یک ساعت پیش
        # بایننس ۲ms پس از ساعت، کوکوین سرِ ساعت — دو سریِ جدا که نباید آمیخته شوند
        self.binance = [[start + k * H8 + 2, float(1e-4 * np.sin(k))] for k in range(1501)]
        self.kucoin = [[start + k * H8, float(2e-4 * np.cos(k))] for k in range(1501)]
        self.path = os.path.join(self.tmp.name, "funding_BTCUSDT.json")
        self.pats = [mock.patch.object(market, "HIST_DIR", self.tmp.name),
                     mock.patch.object(market, "FUNDING_PAGE_GAP", {}),
                     mock.patch.object(market.time, "time", lambda: self.now_ms / 1000.0)]
        for p in self.pats:
            p.start()
        market._neg_until.clear()

    def tearDown(self):
        for p in reversed(self.pats):
            p.stop()
        market._neg_until.clear()
        self.tmp.cleanup()

    def _cache(self, newest_age_ms, mtime_age_s):
        """سندِ بایننسی که آخرین تسویه‌اش ``newest_age_ms`` پیش بوده و ``mtime_age_s`` پیش نوشته شده."""
        rows = [r for r in self.binance if r[0] <= self.now_ms - newest_age_ms]
        doc = {"src": "binance", "from_ms": 0, "rows": rows}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        t = self.now_ms / 1000.0 - mtime_age_s
        os.utime(self.path, (t, t))
        return rows

    def _doc(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def test_blocked_source_with_stale_rows_hands_over_to_kucoin(self):
        self._cache(newest_age_ms=20 * 3_600_000, mtime_age_s=market.FUNDING_TTL + 60)
        fapi = mock.Mock(side_effect=RuntimeError("HTTP 451"))
        kc = _FakeKucoinFunding(self.kucoin)
        with mock.patch.object(market, "_fapi_json", fapi), \
                mock.patch.object(market, "_get_json", side_effect=kc):
            rows = market.get_funding_history("BTCUSDT")
        self.assertEqual(fapi.call_count, 1)
        self.assertTrue(kc.calls)
        want_from = self.now_ms - 1000 * H8
        expect = [r for r in self.kucoin if r[0] >= want_from]
        self.assertEqual(rows, expect[-1000:])
        doc = self._doc()
        self.assertEqual(doc["src"], "kucoin")
        self.assertEqual(doc["from_ms"], want_from)
        self.assertEqual(doc["rows"], expect)                       # فقط ردیف‌های کوکوین، نه آمیخته
        self.assertFalse(market._neg_hit(self.path))
        # بارِ بعد از خودِ کوکوین تازه می‌شود، نه دوباره از بایننسِ مسدود
        self.now_ms += H8 + 120_000
        with mock.patch.object(market, "_fapi_json", side_effect=AssertionError("بایننس")), \
                mock.patch.object(market, "_get_json", side_effect=_FakeKucoinFunding(self.kucoin)):
            market.get_funding_history("BTCUSDT")
        self.assertEqual(self._doc()["src"], "kucoin")

    def test_fresh_rows_keep_the_cached_source_and_only_back_off(self):
        old = self._cache(newest_age_ms=5 * 3_600_000, mtime_age_s=600)   # آخرین تسویه ۹ ساعت پیش
        self.assertLess(self.now_ms - old[-1][0], 10 * 3_600_000)   # موعد گذشته، هنوز کهنه نیست
        fapi = mock.Mock(side_effect=RuntimeError("HTTP 451"))
        other = mock.Mock(side_effect=RuntimeError("نباید منبعِ دیگری خوانده شود"))
        with mock.patch.object(market, "_fapi_json", fapi), \
                mock.patch.object(market, "_get_json", other):
            rows = market.get_funding_history("BTCUSDT")
        self.assertEqual(fapi.call_count, 1)
        self.assertEqual(other.call_count, 0)
        self.assertEqual(rows, old[-1000:])
        self.assertEqual(self._doc()["src"], "binance")
        self.assertTrue(market._neg_hit(self.path))

    def test_source_that_answers_but_stopped_publishing_hands_over(self):
        self._cache(newest_age_ms=20 * 3_600_000, mtime_age_s=market.FUNDING_TTL + 60)
        stopped = [r for r in self.binance if r[0] <= self.now_ms - 20 * 3_600_000]
        with mock.patch.object(market, "_fapi_json", side_effect=_FakeBinanceFunding(stopped)), \
                mock.patch.object(market, "_get_json", side_effect=_FakeKucoinFunding(self.kucoin)):
            rows = market.get_funding_history("BTCUSDT")
        self.assertEqual(rows[-1], self.kucoin[-1])
        self.assertEqual(self._doc()["src"], "kucoin")

    def test_source_answering_no_contract_with_stale_rows_hands_over(self):
        import httpx
        self._cache(newest_age_ms=20 * 3_600_000, mtime_age_s=market.FUNDING_TTL + 60)
        req = httpx.Request("GET", "https://fapi.example/fapi/v1/fundingRate")
        gone = httpx.HTTPStatusError("400", request=req, response=httpx.Response(400, request=req))
        with mock.patch.object(market, "_fapi_json", side_effect=gone), \
                mock.patch.object(market, "_get_json", side_effect=_FakeKucoinFunding(self.kucoin)):
            rows = market.get_funding_history("BTCUSDT")
        self.assertEqual(rows[-1], self.kucoin[-1])
        self.assertEqual(self._doc()["src"], "kucoin")

    def test_no_fresher_source_keeps_the_old_rows_and_backs_off(self):
        old = self._cache(newest_age_ms=20 * 3_600_000, mtime_age_s=market.FUNDING_TTL + 60)
        down = mock.Mock(side_effect=RuntimeError("down"))
        with mock.patch.object(market, "_fapi_json", down), \
                mock.patch.object(market, "_get_json", down):
            rows = market.get_funding_history("BTCUSDT")
            self.assertEqual(rows, old[-1000:])
            n = down.call_count
            self.assertEqual(n, 3)                                     # بایننس، کوکوین، OKX
            market.get_funding_history("BTCUSDT")                      # کشِ منفی: بی شبکه
            self.assertEqual(down.call_count, n)
        self.assertEqual(self._doc()["src"], "binance")

    def test_cache_writes_use_a_per_process_temp_name(self):
        seen = []
        real = os.replace

        def rec(a, b):
            seen.append(a)
            real(a, b)
        with mock.patch.object(market.os, "replace", side_effect=rec):
            market._write_funding_doc(self.path, {"src": "binance", "from_ms": 0, "rows": []})
        self.assertEqual(len(seen), 1)
        self.assertIn(str(os.getpid()), os.path.basename(seen[0]))
        self.assertNotEqual(seen[0], self.path + ".tmp")
        self.assertEqual(self._doc()["src"], "binance")


class LowCoverageFeatureTests(unittest.TestCase):
    """calib-F4: «۰ = نبود» با پوششِ ناچیز در آموزش و اجرا صفر؛ ثابت‌های طبیعی «مرده» نیستند."""

    def test_rare_context_feature_is_zeroed_in_live(self):
        n = len(calib.engine.FEATS)
        j = calib.engine.FEATS.index("funding_dir")
        rows = [[0.1 * ((i + k) % 7) for k in range(n)] for i in range(400)]
        for i, r in enumerate(rows):
            r[j] = 0.3 if i < 8 else 0.0                                   # پوشش ۲٪
        rep = features.health(rows)
        self.assertIn("funding_dir", features.live_zeroed(rep))
        blob = {"live_zeroed": {"events": ["funding_dir"], "dense": []}}
        vec = [0.5] * n
        self.assertEqual(calib._live_feats(blob, vec, "events")[j], 0.0)
        self.assertEqual(calib._live_feats(blob, vec, "dense")[j], 0.5)

    def test_missing_higher_timeframe_is_zeroed_like_other_context(self):
        self.assertIn("htf_align", features.MISSING_AS_ZERO)
        n = len(calib.engine.FEATS)
        rows = [[0.0 if name == "htf_align" else 0.01 * ((i * 7 + k) % 13)
                 for k, name in enumerate(calib.engine.FEATS)] for i in range(100)]
        self.assertIn("htf_align", features.live_zeroed(features.health(rows)))
        self.assertEqual(len(rows[0]), n)

    def test_expected_constants_are_not_reported_dead(self):
        n = len(calib.engine.FEATS)
        rows = [[0.0 if name in ("hour_sin", "rs_dir", "breadth_dir") else 1.0 if name == "hour_cos"
                 else 0.01 * ((i * 7 + k) % 13) for k, name in enumerate(calib.engine.FEATS)]
                for i in range(200)]
        rep = features.health(rows, expected=features.expected_constant("1d"))
        self.assertEqual(rep["dead"], [])
        self.assertTrue(rep["ok"])
        self.assertIn("hour_cos", rep["expected_constant"])
        rep_4h = features.health(rows, expected=features.expected_constant("4h"))
        self.assertIn("hour_sin", rep_4h["dead"])
        self.assertEqual(len(rows[0]), n)


def _basket_hists(n=1500, bar_ms=3_600_000, start_bar=480_000):
    return {s: _klines(n, seed=40 + k, start_bar=start_bar, bar_ms=bar_ms)
            for k, s in enumerate(calib.MARKET_BASKET)}


class MarketStateParityTests(unittest.TestCase):
    """calib-F1: وضعیتِ بازار با یک کد و یک سبد، در یک کندلِ مشخص، در آموزش و اجرا."""

    def test_basket_is_the_live_watchlist(self):
        import watchlist
        self.assertEqual(calib.MARKET_BASKET, tuple(watchlist.SYMBOLS))
        self.assertIn("BTCUSDT", calib.MARKET_BASKET)
        self.assertIn("ETHUSDT", calib.MARKET_BASKET)

    def test_420_live_bars_give_the_training_value_at_that_bar(self):
        full = _basket_hists()
        maps = calib.market_state_maps(full)
        self.assertTrue(maps["dom"] and maps["ethbtc"])
        for end in (900, 1200, 1500):
            live = {s: {k: v[end - 420:end] for k, v in kl.items()} for s, kl in full.items()}
            tk = live["BTCUSDT"]["t"][-1]
            st = calib.market_state_at(live, tk)
            self.assertAlmostEqual(st["dom"], maps["dom"][tk], places=12)
            self.assertAlmostEqual(st["ethbtc"], maps["ethbtc"][tk], places=12)
            self.assertEqual(st["breadth"], 0.0)

    def test_a_coin_missing_the_bar_gives_neutral_dom_not_a_pinned_one(self):
        full = _basket_hists()
        live = {s: {k: v[1080:1500] for k, v in kl.items()} for s, kl in full.items()}
        tk = live["BTCUSDT"]["t"][-1]
        stale = dict(live, SOLUSDT={k: v[:-1] for k, v in live["SOLUSDT"].items()})
        self.assertEqual(calib.market_state_at(stale, tk)["dom"], 0.0)
        # سبدِ کامل مقدارِ واقعی دارد و بیشترِ کندل‌ها روی ±۱ نیستند
        dom = calib.market_state_maps(full)["dom"]
        vals = [dom[t] for t in full["BTCUSDT"]["t"][600:]]
        self.assertLess(np.mean(np.abs(vals) >= 0.999), 0.5)


class LiveMarketStateTests(unittest.TestCase):
    """main._market_state: کندل‌های تازهٔ سبد در کندلِ tk، کش با کلیدِ (tf, tk)."""

    def setUp(self):
        import main
        self.main = main
        self.full = _basket_hists(n=900)
        self.tk = self.full["BTCUSDT"]["t"][-1]
        main._mstate_cache.clear()

    def test_stale_cached_coin_is_refreshed_and_the_state_is_cached_per_bar(self):
        stale_eth = {k: v[:-1] for k, v in self.full["ETHUSDT"].items()}
        calls = []

        def get_klines(s, tf, *a, **k):
            calls.append(s)
            return stale_eth if s == "ETHUSDT" else self.full[s]
        with mock.patch.object(market, "get_klines", side_effect=get_klines), \
                mock.patch.object(market, "_fetch_klines",
                                  side_effect=lambda s, tf, limit: self.full[s]) as fetch:
            st = self.main._market_state("1h", self.tk)
            again = self.main._market_state("1h", self.tk)
        fetch.assert_called_once()
        self.assertEqual(st, again)
        self.assertEqual(len(calls), len(calib.MARKET_BASKET))           # بارِ دوم از کش
        ref = calib.market_state_at(self.full, self.tk)
        self.assertAlmostEqual(st["dom"], ref["dom"], places=12)
        self.assertAlmostEqual(st["ethbtc"], ref["ethbtc"], places=12)

    def test_relative_strength_and_breadth_are_neutral_like_training(self):
        self.assertEqual(self.main._rs_rank("1h").get("BTCUSDT", 0.5), 0.5)
        self.assertIn("rs_dir", features.DISABLED)
        self.assertIn("breadth_dir", features.DISABLED)
        with mock.patch.object(market, "get_klines", side_effect=lambda s, tf, *a, **k: self.full[s]):
            self.assertEqual(self.main._market_state("1h", self.tk)["breadth"], 0.0)
        self.assertEqual(self.main._market_state("5m", self.tk), {"breadth": 0.0, "dom": 0.0, "ethbtc": 0.0})


class LiveGoldTests(unittest.TestCase):
    """calib-F7: طلای زنده در کندلِ تحلیل‌شده، با تازگیِ کندلِ بسته — نه کشِ ۲۴ساعته."""

    def setUp(self):
        import main
        self.main = main
        main._macro_cache.clear()
        self.paxg = _klines(3000, seed=77, start_bar=480_000)

    def test_gold_is_the_training_value_at_the_analysed_bar(self):
        train_map = calib.mom_norm_map(self.paxg["t"], self.paxg["c"])
        live = {k: v[-420:] for k, v in self.paxg.items()}
        tk = live["t"][-1]
        with mock.patch.object(market, "get_klines", return_value=live), \
                mock.patch.object(market, "get_history", side_effect=AssertionError("کشِ ۲۴ساعته")):
            g = self.main._macro("1h", tk)["gold"]
        self.assertAlmostEqual(g, train_map[tk], places=12)

    def test_a_grace_stale_paxg_cache_is_refreshed_for_the_new_bar(self):
        live = {k: v[-420:] for k, v in self.paxg.items()}
        stale = {k: v[:-1] for k, v in live.items()}
        tk = live["t"][-1]
        with mock.patch.object(market, "get_klines", return_value=stale), \
                mock.patch.object(market, "_fetch_klines", return_value=live) as fetch:
            g1 = self.main._macro("1h", tk)
            g2 = self.main._macro("1h", tk)
        fetch.assert_called_once()
        self.assertEqual(g1, g2)
        self.assertNotEqual(g1["gold"], 0.0)

    def test_missing_bar_gives_neutral_gold(self):
        live = {k: v[-420:] for k, v in self.paxg.items()}
        stale = {k: v[:-1] for k, v in live.items()}
        with mock.patch.object(market, "get_klines", return_value=stale), \
                mock.patch.object(market, "_fetch_klines", return_value=stale):
            self.assertEqual(self.main._macro("1h", live["t"][-1]), {"gold": 0.0})


class HtfSignRawZTests(unittest.TestCase):
    """calib-F10: htf_sign با zِ خام و همان تابعِ آموزش، نه zِ گردشدهٔ خروجیِ تحلیل."""

    def setUp(self):
        import main
        self.main = main
        main._htf_z_cache.clear()
        self.hk = _klines(420, seed=8, start_bar=120_000, bar_ms=calib.TF_MS["4h"])

    def _run(self, z_at_t0, tk):
        z = np.zeros(len(self.hk["t"]))
        t0 = (tk // calib.TF_MS["4h"]) * calib.TF_MS["4h"] - calib.TF_MS["4h"]
        z[self.hk["t"].index(t0)] = z_at_t0
        with mock.patch.object(market, "get_klines", return_value=self.hk), \
                mock.patch.object(self.main.engine, "component_series", return_value={"z": z}):
            return self.main._htf_sign_live("AUSDT", "4h", tk)

    def test_z_just_above_the_threshold_is_not_rounded_away(self):
        tk = self.hk["t"][-1] + calib.TF_MS["4h"] + 3 * calib.TF_MS["1h"]   # 1hِ آخرِ سطلِ بعدی
        self.assertEqual(round(0.3021, 2) > 0.3, False)                   # رفتارِ قبلی: ۰
        self.assertEqual(self._run(0.3021, tk), 1)
        self.main._htf_z_cache.clear()
        self.assertEqual(self._run(-0.3021, tk), -1)

    def test_same_function_as_training(self):
        tk = self.hk["t"][-1] + calib.TF_MS["4h"]
        cs = calib.engine.component_series(*[np.asarray(self.hk[k], float) for k in "ohlcv"])
        zmap = {t: float(z) for t, z in zip(self.hk["t"], cs["z"])}
        with mock.patch.object(market, "get_klines", return_value=self.hk):
            self.assertEqual(self.main._htf_sign_live("AUSDT", "4h", tk),
                             calib._htf_sign(zmap, tk, "4h"))


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


class BuildFeatureHealthTests(unittest.TestCase):
    def test_build_records_and_applies_the_live_zeroed_features(self):
        h = _BuildHarness(prereg=None)
        table = h.run()
        blob = table["tfs"]["4h"]
        self.assertIn("funding_dir", blob["live_zeroed"]["events"])    # بی تاریخچهٔ فاندینگ
        self.assertIn("funding_dir", blob["live_zeroed"]["dense"])
        self.assertIn("feature_health_dense", blob)
        self.assertNotIn("rs_dir", blob["feature_health"]["dead"])       # خنثیِ عمدی، نه «مرده»
        j = calib.engine.FEATS.index("funding_dir")
        self.assertTrue(all(e["feats"][j] == 0.0 for e in h.seen["model"][0]))

    def test_training_market_state_comes_from_the_basket_only(self):
        h = _BuildHarness(prereg=None)
        with mock.patch.object(calib, "market_state_maps", wraps=calib.market_state_maps) as spy:
            h.run()
        hists = spy.call_args[0][0]
        self.assertEqual(set(hists), {"BTCUSDT", "ETHUSDT"})             # XUSDT بیرونِ سبد است
        self.assertTrue(set(hists) <= set(calib.MARKET_BASKET))
        j = calib.engine.FEATS.index("rs_dir")
        self.assertTrue(all(e["feats"][j] == 0.0 for e in h.seen["model"][0]))


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
