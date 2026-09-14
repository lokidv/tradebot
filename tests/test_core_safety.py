import os
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import autotrader  # noqa: E402
import broker  # noqa: E402
import calib  # noqa: E402
import engine  # noqa: E402
import paper  # noqa: E402
import shadow  # noqa: E402


class ValidationSafetyTests(unittest.TestCase):
    def test_walk_forward_groups_timestamp_and_purges_label_horizon(self):
        ts = np.repeat(np.arange(100, dtype=np.int64) * 1_000, 4)
        splits = calib._walk_forward_splits(ts, embargo_ms=5_000)
        self.assertEqual(len(splits), calib.WF_FOLDS)
        for train_idx, test_idx in splits:
            train_ts, test_ts = ts[train_idx], ts[test_idx]
            self.assertTrue(set(train_ts).isdisjoint(set(test_ts)))
            self.assertLess(train_ts.max(), test_ts.min() - 5_000)

    def test_model_requires_probability_skill_not_only_ranking_lift(self):
        no_skill = {
            "n_oos": 200, "oos_lift": 12.0, "oos_base": 40.0,
            "oos_brier": 0.24,
        }
        useful = dict(no_skill, oos_brier=0.22)
        self.assertFalse(calib._model_trusted(no_skill, calib.MIN_SETUP_LIFT))
        self.assertTrue(calib._model_trusted(useful, calib.MIN_SETUP_LIFT))

    def test_grouped_walk_forward_produces_oos_predictions_without_worker_pool(self):
        rng = np.random.RandomState(3)
        x = rng.normal(size=(400, 5))
        y = (x[:, 0] > 0).astype(float)
        ts = np.arange(400, dtype=np.int64) * 3_600_000
        with mock.patch.object(calib, "_submit", return_value=None):
            pred = calib._walk_forward(
                x, y, "logit", ts=ts, embargo_ms=8 * 3_600_000,
            )
        self.assertGreater(np.sum(~np.isnan(pred)), 150)

    def test_probability_prediction_exposes_conservative_interval(self):
        meta = {"kind": "logit", "n_oos": 100, "oos_brier": 0.20}
        with mock.patch.object(calib, "_score_model", return_value=0.60):
            result = calib._score_model_details(meta, [0.0, 1.0])
        self.assertLess(result["p_low"], result["p"])
        self.assertGreater(result["p_high"], result["p"])
        self.assertGreater(result["p_uncertainty"], 0)

    def test_direct_net_return_model_must_pass_temporal_holdout(self):
        rng = np.random.RandomState(17)
        events = []
        for i in range(900):
            x = rng.normal(size=6)
            risk_pct = 1.0
            net_r = 0.62 * x[0] + 0.12 * rng.normal()
            gross_r = net_r + calib.COST_PCT / risk_pct
            events.append({
                "ts": i * calib.TF_MS["1h"],
                "feats": x.tolist(),
                "risk_pct": risk_pct,
                "r": gross_r,
                "regime": "trend",
            })
        model = calib._fit_edge_model(events, "1h", calib.COST_PCT)
        self.assertIsNotNone(model)
        self.assertTrue(calib._edge_model_trusted(model))
        scored = calib._score_edge(
            model, [2.0, 0, 0, 0, 0, 0], 1.0, calib.COST_PCT, "trend")
        self.assertGreater(scored["edge_lcb_r"], 0.03)
        self.assertTrue(scored["regime_ok"])

    def test_final_policy_is_selected_before_untouched_temporal_test(self):
        rng = np.random.RandomState(29)
        events = []
        for i in range(1_200):
            x = rng.normal(size=6)
            net_r = 0.85 * x[0] + 0.10 * rng.normal()
            events.append({
                "ts": i * calib.TF_MS["1h"],
                "feats": x.tolist(),
                "risk_pct": 1.0,
                "r": float(net_r + calib.COST_PCT),
                "regime": "trend",
                "setup": "zx",
            })
        with mock.patch.object(calib, "HAS_LGBM", False), \
                mock.patch.object(calib, "_submit", return_value=None):
            model = calib._fit_policy_model(events, "1h", calib.COST_PCT)
        self.assertIsNotNone(model)
        self.assertTrue(calib._policy_model_trusted(model))
        self.assertGreaterEqual(model["test"]["n"], calib.POLICY_MIN_TEST)
        self.assertGreater(model["test"]["uplift_r"], calib.POLICY_MIN_UPLIFT_R)

    def test_dense_action_policy_can_choose_long_short_or_no_trade(self):
        rng = np.random.RandomState(37)
        dx, dy, dr = [], [], []
        for i in range(4_000):
            x = rng.normal(size=6)
            long_x, short_x = x.copy(), x.copy()
            short_x[0] *= -1
            net_long = 0.72 * long_x[0] + 0.10 * rng.normal()
            net_short = 0.72 * short_x[0] + 0.10 * rng.normal()
            dx.append((long_x.tolist(), short_x.tolist()))
            dy.append(((i // 20) * calib.TF_MS["1h"], float(net_long > net_short)))
            dr.append((float(net_long + calib.COST_PCT),
                       float(net_short + calib.COST_PCT), 1.0))
        with mock.patch.object(calib, "HAS_LGBM", False):
            model = calib._fit_action_policy(dx, dy, dr, "1h", calib.COST_PCT)
        self.assertIsNotNone(model)
        self.assertTrue(calib._action_policy_trusted(model))
        scored = calib._score_action_policy(
            model, [2, 0, 0, 0, 0, 0], [-2, 0, 0, 0, 0, 0],
            1.0, calib.COST_PCT,
        )
        self.assertEqual(scored["side"], "long")
        self.assertTrue(scored["policy_pass"])


class RiskPolicyTests(unittest.TestCase):
    def test_activity_level_never_forces_positions_or_unbounded_risk(self):
        for level in range(1, 11):
            params = autotrader.level_params(level)
            self.assertEqual(params["min_open"], 0)
            self.assertFalse(params["allow_lean"])
            self.assertLessEqual(params["max_open"], 5)
            self.assertLessEqual(params["risk_pct_equity"], 0.5)
            self.assertLessEqual(params["heat_cap"], 2.8)
            self.assertNotIn("1d", params["scan_tfs"])
            self.assertNotIn("15m", params["scan_tfs"])

    def test_pending_orders_reserve_directional_risk(self):
        order = {
            "BTCUSDT": {"side": "long", "size": 1_000, "r_abs": 2.0, "target": 100.0},
            "ETHUSDT": {"side": "short", "size": 500, "r_abs": 5.0, "target": 100.0},
        }
        with mock.patch.dict(autotrader._pending, order, clear=True):
            self.assertAlmostEqual(autotrader._pending_risk_usd(), 45.0)
            self.assertAlmostEqual(autotrader._pending_risk_usd("long"), 20.0)
            self.assertAlmostEqual(
                autotrader._pending_risk_usd(exclude_symbol="BTCUSDT"), 25.0)

    def test_testnet_quantity_is_not_multiplied_by_leverage(self):
        class FakeBinance(broker.BinanceTestnet):
            def __init__(self):
                self.leverage = 5
                self.calls = []

            def _sym_info(self, symbol):
                return 0.001, 0.01, 0.001, 5.0

            def mark_price(self, symbol):
                return 100.0

            def _ensure_leverage(self, symbol):
                return None

            def _signed(self, method, path, **params):
                self.calls.append((method, path, params))
                if params.get("type") == "MARKET":
                    return {"orderId": 7, "avgPrice": "100.1"}
                return {}

            def close_symbol(self, symbol):
                return True

        client = FakeBinance()
        result = client.open_bracket("BTCUSDT", "long", 1_000, 95, 110)
        market_call = next(c for c in client.calls if c[2].get("type") == "MARKET")
        self.assertAlmostEqual(market_call[2]["quantity"], 10.0)
        self.assertAlmostEqual(result["price"], 100.1)
        self.assertAlmostEqual(result["sl"], 95.1)
        self.assertAlmostEqual(result["tp"], 110.1)


class DecisionPolicyTests(unittest.TestCase):
    @staticmethod
    def _klines():
        n = 420
        c = 100 + np.linspace(0, 4, n) + 0.3 * np.sin(np.arange(n) / 7)
        return {
            "t": (np.arange(n, dtype=np.int64) * 3_600_000).tolist(),
            "o": c.tolist(), "h": (c + 0.4).tolist(),
            "l": (c - 0.4).tolist(), "c": c.tolist(),
            "v": np.full(n, 1_000.0).tolist(),
        }

    def test_positive_point_estimate_cannot_override_untrusted_final_policy(self):
        suggestion = {
            "side": "long", "entry": 104.0, "sl": 103.0, "tp": 105.8,
            "risk_pct": 0.96, "rr": 1.8, "grade": "A", "viable": True,
            "tradeable": True, "status": "آماده", "reasons": [], "time_stop_min": 2_400,
        }

        def fake_predict(_tf, _feats, _risk, **_kwargs):
            return {
                "source": "policy", "policy_trusted": False, "policy_pass": True,
                "policy_score": 2.0, "policy_margin": 1.0,
                "p_win": 58.0, "p_win_low": 49.0,
                "p_win_high": 67.0, "p_uncertainty": 9.0,
                "n": 300, "avg_r": 0.30, "ev_pct": 0.25, "ev_lcb_pct": -0.04,
                "reliability": "خوب", "oos_lift": 9.0, "base": 40.0,
                "edge_trusted": True, "edge_r": 0.35, "edge_lcb_r": 0.12,
                "edge_uncertainty_r": 0.23, "edge_rank_ic": 0.08,
                "edge_lift_r": 0.2, "feature_zmax": 1.0,
                "regime_ok": True, "regime_stats": {
                    "selected_n": 40, "selected_avg_net_r": 0.2,
                },
            }

        with mock.patch.object(engine, "trade_suggestion", return_value=suggestion), \
                mock.patch.object(engine, "quick_backtest",
                                  return_value={"n": 0, "win_rate": None, "avg_r": None,
                                                "timeouts": 0, "profit_factor": None}):
            result = engine.analyze(self._klines(), "1h", predict_fn=fake_predict)
        self.assertFalse(result["trade"]["tradeable"])
        self.assertIn("سیاست نهایی", result["trade"]["status"])


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_path = paper.PATH
        paper.PATH = os.path.join(self.tmp.name, "positions.json")

    def tearDown(self):
        paper.PATH = self.old_path
        self.tmp.cleanup()

    def test_cost_and_stop_gap_are_included(self):
        paper.open_position(
            "BTCUSDT", "1h", "long", 100, 90, 120, 1_000, 60,
            cost_pct=0.2,
        )
        db = paper.refresh({"BTCUSDT": 85})
        closed = db["closed"][0]
        self.assertEqual(closed["exit_price"], 85)
        self.assertAlmostEqual(closed["gross_pnl_pct"], -15.0)
        self.assertAlmostEqual(closed["pnl_pct"], -15.2)
        self.assertAlmostEqual(closed["pnl_usdt"], -152.0)

    def test_quick_backtest_counts_timeout_and_cost(self):
        n = 270
        o = np.full(n, 100.0)
        h = np.full(n, 100.1)
        l = np.full(n, 99.9)
        c = np.full(n, 100.0)
        cs = {"a14": np.full(n, 1.0)}

        def one_signal(_cs, _o, _h, _l, _c, i):
            return (1, "zx") if i == 220 else (0, None)

        with mock.patch.object(engine, "setup_signal", side_effect=one_signal):
            result = engine.quick_backtest(o, h, l, c, cs, "1h", cost_pct=0.15)
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["timeouts"], 1)
        self.assertLess(result["avg_r"], 0)

    def test_shadow_result_is_net_of_round_trip_cost(self):
        old_path = shadow.SHADOW_PATH
        shadow.SHADOW_PATH = os.path.join(self.tmp.name, "signals.json")
        try:
            candle_ts = int(time.time() * 1000) - 10_000
            shadow.log_signal(
                "BTCUSDT", "1h", "long", 100, 99, 102, "zx",
                55, 0.4, candle_ts, 60, cost_pct=0.20,
            )
            kl = {
                "t": [candle_ts, candle_ts + 1_000],
                "o": [100, 100], "h": [100, 102.1],
                "l": [100, 99.5], "c": [100, 102],
                "v": [1, 1],
            }
            shadow.resolve(lambda _symbol, _tf: kl)
            result = shadow._load()["resolved"][0]
            self.assertAlmostEqual(result["gross_r"], 2.0)
            self.assertAlmostEqual(result["r_mult"], 1.8)
            self.assertAlmostEqual(shadow.stats("1h")["by_setup"]["zx"]["avg_r"], 1.8)
        finally:
            shadow.SHADOW_PATH = old_path


if __name__ == "__main__":
    unittest.main()
