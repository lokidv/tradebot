# -*- coding: utf-8 -*-
"""تست‌های پذیرشِ فاز ۰ — قفلِ ایمنیِ سرمایه.

این تست‌ها تضمین می‌کنند که هیچ مسیری نمی‌تواند بدون ترکیبِ پیش‌ثبت‌شده در
``gates.json`` معامله‌ای را قابل‌اجرا کند، «جیب زنده» دیگر مرجع نیست، بروکرِ
غیرِ تست‌نت بدون مجوز ساخته نمی‌شود، و سیاستِ مردود عددِ گمراه‌کننده پخش نمی‌کند.
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

import broker  # noqa: E402
import calib  # noqa: E402
import engine  # noqa: E402
import gates  # noqa: E402
import meta_gate  # noqa: E402


def _write_gates(path, **over):
    blob = dict(gates.DEFAULTS)
    blob.update(over)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f)


class _GatesFileMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "gates.json")
        self._old_path = gates.GATES_PATH
        gates.GATES_PATH = self.path
        _write_gates(self.path)
        gates.load_gates(force=True)

    def tearDown(self):
        gates.GATES_PATH = self._old_path
        gates._cache.update(mtime=None, data=None)
        self.tmp.cleanup()

    def set_combos(self, combos):
        _write_gates(self.path, allowed_combos=list(combos))
        gates.load_gates(force=True)


class GateFileTests(_GatesFileMixin, unittest.TestCase):
    def test_missing_file_is_fail_closed(self):
        gates.GATES_PATH = os.path.join(self.tmp.name, "does_not_exist.json")
        gates._cache.update(mtime=None, data=None)
        self.assertEqual(gates.allowed_combos(), [])
        self.assertFalse(gates.live_allowed())
        self.assertFalse(gates.is_combo_allowed("1h", "zx", "long"))

    def test_only_listed_combo_is_allowed(self):
        self.assertFalse(gates.is_combo_allowed("1h", "zx", "long"))
        self.set_combos(["1h|zx|long"])
        self.assertTrue(gates.is_combo_allowed("1h", "zx", "long"))
        self.assertFalse(gates.is_combo_allowed("1h", "zx", "short"))
        self.assertFalse(gates.is_combo_allowed("4h", "zx", "long"))

    def test_live_needs_both_flag_and_matching_env_ack(self):
        _write_gates(self.path, live_allowed=True, version=7)
        gates.load_gates(force=True)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(gates.LIVE_ACK_ENV, None)
            self.assertFalse(gates.live_allowed())
            os.environ[gates.LIVE_ACK_ENV] = "6"
            self.assertFalse(gates.live_allowed())
            os.environ[gates.LIVE_ACK_ENV] = "7"
            self.assertTrue(gates.live_allowed())

    def test_allowed_combos_cannot_be_written_without_judge_token(self):
        with self.assertRaises(gates.GateError):
            gates.set_allowed_combos(["1h|zx|long"], "manual", judge_token=None)
        self.assertEqual(gates.allowed_combos(), [])

    def test_unsafe_risk_caps_are_rejected(self):
        _write_gates(self.path, max_risk_pct_per_trade=25.0)
        with self.assertRaises(gates.GateError):
            gates.load_gates(force=True)


class BrokerLockdownTests(_GatesFileMixin, unittest.TestCase):
    def test_real_exchange_broker_is_refused_while_live_is_off(self):
        with self.assertRaises(gates.GateError):
            broker.make_broker({"broker": "binance", "api_key": "k", "api_secret": "s"})

    def test_local_and_testnet_remain_available(self):
        self.assertIsNone(broker.make_broker({"broker": "local"}))
        bk = broker.make_broker({"broker": "binance_testnet",
                                 "api_key": "k", "api_secret": "s", "leverage": 2})
        self.assertIsInstance(bk, broker.BinanceTestnet)

    def test_real_exchange_still_unavailable_even_when_live_is_acknowledged(self):
        _write_gates(self.path, live_allowed=True, version=3)
        gates.load_gates(force=True)
        with mock.patch.dict(os.environ, {gates.LIVE_ACK_ENV: "3"}):
            with self.assertRaises(NotImplementedError):
                broker.make_broker({"broker": "binance"})


class _AnalyzeHarness(_GatesFileMixin):
    """موتور را با یک ستاپ zx تأییدشده و سیاستِ معتبر اجرا می‌کند تا فقط گیت سنجیده شود."""

    TF = "1h"

    @staticmethod
    def _klines(n=420):
        c = 100 + np.linspace(0, 4, n) + 0.3 * np.sin(np.arange(n) / 7)
        return {
            "t": (np.arange(n, dtype=np.int64) * 3_600_000).tolist(),
            "o": c.tolist(), "h": (c + 0.4).tolist(),
            "l": (c - 0.4).tolist(), "c": c.tolist(),
            "v": np.full(n, 1_000.0).tolist(),
        }

    @staticmethod
    def _suggestion():
        return {
            "side": "long", "entry": 104.0, "sl": 103.0, "tp": 105.8,
            "risk_pct": 0.96, "rr": 1.8, "grade": "A", "viable": True,
            "tradeable": True, "status": "آماده", "reasons": [], "time_stop_min": 2_400,
        }

    @staticmethod
    def _trusted_policy(*_a, **_k):
        return {
            "source": "policy", "policy_trusted": True, "policy_pass": True,
            "policy_score": 2.0, "policy_margin": 1.0, "policy_test": {"n": 400},
            "p_win": 58.0, "p_win_low": 49.0, "p_win_high": 67.0, "p_uncertainty": 9.0,
            "n": 400, "avg_r": 0.30, "ev_pct": 0.25, "ev_lcb_pct": 0.10,
            "reliability": "خوب", "oos_lift": 9.0, "base": 40.0,
            "edge_trusted": True, "edge_r": 0.35, "edge_lcb_r": 0.12,
            "edge_uncertainty_r": 0.23, "edge_rank_ic": 0.08, "edge_lift_r": 0.2,
            "feature_zmax": 1.0, "regime_veto": False, "regime_ok": True,
            "regime_stats": {"selected_n": 40, "selected_avg_net_r": 0.2},
            "authority": "policy",
        }

    def run_analyze(self, predict_fn=None, extras=None):
        def one_setup(_cs, _o, _h, _l, _c, i):
            return (1, "zx") if i >= 419 else (0, None)

        approve = {"approve": True, "score": 0.8, "size_mult": 1.0,
                   "reasons": [], "blocks": [], "regime": "chop",
                   "confluence": 80, "session_tier": "high", "oi_z": None}
        with mock.patch.object(engine, "setup_signal", side_effect=one_setup), \
                mock.patch.object(engine, "trade_suggestion", return_value=self._suggestion()), \
                mock.patch.object(engine, "quick_backtest",
                                  return_value={"n": 0, "win_rate": None, "avg_r": None,
                                                "timeouts": 0, "profit_factor": None}), \
                mock.patch.object(meta_gate, "evaluate", return_value=approve):
            return engine.analyze(self._klines(), self.TF,
                                  predict_fn=predict_fn or self._trusted_policy,
                                  extras=extras or {})


class GateBlocksTradingTests(_AnalyzeHarness, unittest.TestCase):
    def test_trusted_policy_is_still_blocked_when_combo_is_not_preregistered(self):
        tr = self.run_analyze()["trade"]
        self.assertFalse(tr["gate_allowed"])
        self.assertFalse(tr["tradeable"])
        self.assertIn("قفل ایمنی", tr["status"])

    def test_trusted_policy_trades_only_after_the_combo_is_preregistered(self):
        self.set_combos(["1h|zx|long"])
        tr = self.run_analyze()["trade"]
        self.assertTrue(tr["gate_allowed"])
        self.assertTrue(tr["tradeable"])
        self.assertEqual(tr["authority"], "policy")

    def test_gate_is_side_specific(self):
        self.set_combos(["1h|zx|short"])
        tr = self.run_analyze()["trade"]
        self.assertFalse(tr["gate_allowed"])
        self.assertFalse(tr["tradeable"])


class PocketIsNoLongerAnAuthorityTests(_AnalyzeHarness, unittest.TestCase):
    POCKET = {
        "1h|zx|long": {"n": 18, "avg_r": 0.5, "win_rate": 61.0, "lcb_r": 0.3,
                       "alive": True, "grade": "A", "size_hint": 1.0,
                       "tf": "1h", "setup": "zx", "side": "long", "recent": 0.4,
                       "half1": 0.3, "sd": 1.0, "half0": 0.6},
    }

    @staticmethod
    def _rejected_policy(*_a, **_k):
        return {
            "source": "policy", "policy_trusted": False, "policy_pass": False,
            "policy_score": None, "policy_margin": None, "policy_test": {"n": 400},
            "p_win": None, "n": 400, "avg_r": None, "ev_pct": None, "ev_lcb_pct": None,
            "reliability": "ردِ دادگاه", "authority": None,
            "edge_trusted": False, "regime_veto": False, "regime_ok": True,
            "feature_zmax": 1.0,
        }

    def test_live_pocket_never_grants_authority_even_when_combo_is_preregistered(self):
        self.set_combos(["1h|zx|long"])
        tr = self.run_analyze(predict_fn=self._rejected_policy,
                              extras={"edge_pockets": self.POCKET})["trade"]
        self.assertIsNone(tr.get("authority"))
        self.assertFalse(tr["tradeable"])
        self.assertFalse(tr.get("policy_trusted"))
        self.assertTrue(tr.get("pocket_diagnostic_only"))
        self.assertIsNone(tr.get("pocket_size_hint"))

    def test_setup_model_alone_never_grants_authority(self):
        def setup_model_only(*_a, **_k):
            return {
                "source": "model", "policy_trusted": False, "policy_pass": False,
                "p_win": 58.0, "p_win_low": 50.0, "p_win_high": 66.0,
                "p_uncertainty": 8.0, "n": 300, "avg_r": 0.3,
                "ev_pct": 0.4, "ev_lcb_pct": 0.35,      # EV-LCB مثبت — قبلاً مجوز می‌داد
                "reliability": "خوب", "oos_lift": 9.0, "base": 40.0,
                "edge_trusted": False, "authority": None, "feature_zmax": 1.0,
            }

        self.set_combos(["1h|zx|long"])
        tr = self.run_analyze(predict_fn=setup_model_only)["trade"]
        self.assertIsNone(tr.get("authority"))
        self.assertFalse(tr["tradeable"])


class RejectedPolicyPublishesNoNumbersTests(unittest.TestCase):
    """سیاستِ مردود با نرمال‌سازیِ مخرب (edge_sd≈۰) امتیازهای ±۱۰⁵ می‌ساخت."""

    TEST_STATS = {"n": 1706, "avg_net_r": -0.11, "uplift_r": 0.0}
    PM = {"test": TEST_STATS, "base_win": 34.06}

    @classmethod
    def _degenerate_score(cls, cost_pct):
        # همان چیزی که _score_policy روی مدلِ شیب‌صفرِ 4h تولید می‌کند
        blown_up = -(cost_pct - 0.15) * 1e6 / 1.5
        return {
            "p_win": 33.9, "p_win_low": 25.9, "p_win_high": 41.9, "p_uncertainty": 8.0,
            "edge_r": -0.11 - (cost_pct - 0.15) / 1.5,
            "edge_lcb_r": -0.19, "edge_uncertainty_r": 0.08,
            "policy_score": blown_up, "policy_margin": blown_up,
            "policy_pass": False, "policy_test": cls.TEST_STATS,
            "feature_zmax": 1.2, "regime_stats": None,
            "regime_veto": False, "regime_ok": True,
        }

    def test_payload_is_identical_across_cost_tiers_and_carries_no_score(self):
        cheap = calib._policy_untrusted_payload(self.PM, self._degenerate_score(0.08), 1.5)
        dear = calib._policy_untrusted_payload(self.PM, self._degenerate_score(0.30), 1.5)
        self.assertEqual(cheap, dear)
        for key in ("policy_score", "policy_margin", "edge_r", "edge_lcb_r",
                    "ev_pct", "ev_lcb_pct", "avg_r", "p_win"):
            self.assertIsNone(cheap[key], f"{key} باید None باشد")
        self.assertFalse(cheap["policy_trusted"])
        self.assertFalse(cheap["policy_pass"])
        self.assertIsNone(cheap["authority"])
        self.assertTrue(cheap["policy_court_failed"])


if __name__ == "__main__":
    unittest.main()
