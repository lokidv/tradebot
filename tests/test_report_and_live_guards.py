# -*- coding: utf-8 -*-
"""تست‌های پذیرشِ فاز ۳ و ۴ — گزارشِ مرحله‌ای، سقف‌های ریسک، کلیدِ قطع، تصمیمِ مقیاس."""
import json
import os
import sys
import tempfile
import time
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import autotrader  # noqa: E402
import candidates  # noqa: E402
import fill_quality  # noqa: E402
import gates  # noqa: E402
import journal  # noqa: E402
import paper  # noqa: E402
import report  # noqa: E402
import research  # noqa: E402

DAY = 86400.0


class _IsolatedData:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self._saved = dict(
            gp=gates.GATES_PATH, jp=journal.PATH, fq=fill_quality.PATH, pp=paper.PATH,
            cp=candidates.CAND_PATH, rp=candidates.RESULT_PATH, hs=report.HEALTH_SAMPLES,
            rd=report.REPORT_DIR, pr=research.PREREG_PATH, fd=research.FINAL_DIR,
            ac=autotrader.CFG_PATH)
        gates.GATES_PATH = os.path.join(d, "gates.json")
        journal.PATH = os.path.join(d, "journal.jsonl")
        fill_quality.PATH = os.path.join(d, "fq.json")
        paper.PATH = os.path.join(d, "positions.json")
        candidates.CAND_PATH = os.path.join(d, "cands.jsonl")
        candidates.RESULT_PATH = os.path.join(d, "results.jsonl")
        report.HEALTH_SAMPLES = os.path.join(d, "health.jsonl")
        report.REPORT_DIR = os.path.join(d, "reports")
        research.PREREG_PATH = os.path.join(d, "prereg.json")
        research.FINAL_DIR = os.path.join(d, "final")
        autotrader.CFG_PATH = os.path.join(d, "autobot.json")
        with open(gates.GATES_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(gates.DEFAULTS), f)
        gates.load_gates(force=True)
        journal._reset_cache()
        candidates._reset_cache()
        autotrader._pending.clear()
        autotrader._cooldown.clear()
        autotrader._risk_off_day = None

    def tearDown(self):
        s = self._saved
        gates.GATES_PATH, journal.PATH, fill_quality.PATH, paper.PATH = s["gp"], s["jp"], s["fq"], s["pp"]
        candidates.CAND_PATH, candidates.RESULT_PATH = s["cp"], s["rp"]
        report.HEALTH_SAMPLES, report.REPORT_DIR = s["hs"], s["rd"]
        research.PREREG_PATH, research.FINAL_DIR = s["pr"], s["fd"]
        autotrader.CFG_PATH = s["ac"]
        gates._cache.update(mtime=None, data=None)
        journal._reset_cache()
        candidates._reset_cache()
        autotrader._pending.clear()
        autotrader._risk_off_day = None
        self.tmp.cleanup()


class RiskCapsComeFromGatesTests(_IsolatedData, unittest.TestCase):
    def test_activity_level_can_never_exceed_the_gate_caps(self):
        for lv in range(1, 11):
            p = autotrader.level_params(lv)
            self.assertLessEqual(p["risk_pct_equity"], gates.DEFAULTS["max_risk_pct_per_trade"])
            self.assertLessEqual(p["heat_cap"], gates.DEFAULTS["heat_cap_pct"])
            self.assertLessEqual(p["max_open"], gates.DEFAULTS["max_open"])
            self.assertEqual(p["daily_loss_halt_pct"], gates.DEFAULTS["daily_loss_halt_pct"])

    def test_tighter_gates_tighten_every_level(self):
        blob = dict(gates.DEFAULTS, max_risk_pct_per_trade=0.1, max_open=1, heat_cap_pct=0.3)
        with open(gates.GATES_PATH, "w", encoding="utf-8") as f:
            json.dump(blob, f)
        gates.load_gates(force=True)
        p = autotrader.level_params(10)
        self.assertEqual(p["risk_pct_equity"], 0.1)
        self.assertEqual(p["max_open"], 1)
        self.assertEqual(p["heat_cap"], 0.3)

    def test_weekly_loss_counts_recent_bot_losses(self):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        db = {"closed": [
            {"opened_by": "bot", "pnl_usdt": -80, "closed_at": (now - timedelta(days=2)).isoformat()},
            {"opened_by": "bot", "pnl_usdt": -50, "closed_at": (now - timedelta(days=20)).isoformat()},
            {"opened_by": "user", "pnl_usdt": -999, "closed_at": now.isoformat()},
        ], "open": [{"opened_by": "bot", "pnl_usdt": -10}]}
        self.assertAlmostEqual(autotrader._weekly_pnl_bot(db), -90.0)


class KillSwitchTests(_IsolatedData, unittest.TestCase):
    def test_kill_flattens_disables_and_empties_the_allow_list(self):
        paper.open_position("BTCUSDT", "4h", "long", 100, 98, 103.6, 500, 9600,
                            opened_by="bot", cost_pct=0.08)
        paper.open_position("ETHUSDT", "4h", "long", 100, 98, 103.6, 500, 9600,
                            opened_by="user", cost_pct=0.08)
        autotrader.save_cfg({"enabled": True, "level": 5})
        research.register({"4h": (0, 10_000_000_000)}, now=time.time())
        gver = gates.load_gates(force=True)["version"]
        gates.set_allowed_combos(["4h|zx|long"], "t", judge_token=f"judge:{gver}",
                                 symbols=["BTCUSDT"])
        res = autotrader.kill_switch("test")
        self.assertEqual(res["closed"], 1)
        db = paper.list_positions()
        self.assertEqual([p["symbol"] for p in db["open"]], ["ETHUSDT"])   # دستیِ کاربر دست نخورد
        self.assertFalse(autotrader.load_cfg()["enabled"])
        self.assertEqual(gates.allowed_combos(), [])
        self.assertFalse(gates.load_gates(force=True)["live_allowed"])

    def test_kill_state_survives_a_restart(self):
        autotrader.kill_switch("test")
        autotrader._risk_off_day = None
        autotrader.restore_state()
        self.assertEqual(autotrader._risk_off_day, time.strftime("%Y-%m-%d", time.gmtime()))

    def test_every_gate_change_is_journaled(self):
        autotrader.kill_switch("x")
        self.assertTrue(journal.events(kinds=[journal.GATE_CHANGE]))


class HealthGateTests(_IsolatedData, unittest.TestCase):
    def test_red_health_blocks_new_entries(self):
        autotrader._fns["health"] = lambda: {"status": "red", "problems": ["مدل کهنه"]}
        try:
            self.assertIn("مدل کهنه", autotrader._health_blocks_entries())
        finally:
            autotrader._fns.pop("health", None)

    def test_green_health_does_not_block(self):
        autotrader._fns["health"] = lambda: {"status": "green", "problems": []}
        try:
            self.assertIsNone(autotrader._health_blocks_entries())
        finally:
            autotrader._fns.pop("health", None)

    def test_green_fraction_over_the_window(self):
        now = time.time()
        for k in range(100):
            report.record_health_sample("green" if k < 97 else "red", now=now - k * 600)
        frac, n = report.health_green_fraction(30, now=now)
        self.assertEqual(n, 100)
        self.assertAlmostEqual(frac, 0.97)
        self.assertFalse(report.evaluate_ops(now=now)["pass"])


class ShadowStageTests(unittest.TestCase):
    def test_short_or_thin_shadow_record_fails_with_reasons(self):
        v = report.evaluate_shadow({"days_observed": 20, "n": 40, "lcb": None,
                                    "mean": 0.3, "win_rate": 45, "brier_skill": 0.02})
        self.assertFalse(v["pass"])
        self.assertTrue(any("روزهای مشاهده" in r for r in v["reasons"]))
        self.assertTrue(any("ردیف" in r for r in v["reasons"]))

    def test_good_shadow_record_passes(self):
        v = report.evaluate_shadow({"days_observed": 95, "n": 260, "lcb": 0.04,
                                    "mean": 0.14, "win_rate": 41.5, "brier_skill": 0.015},
                                   frozen_mean=0.2)
        self.assertTrue(v["pass"], v["reasons"])

    def test_shadow_that_drifts_far_from_the_frozen_test_fails(self):
        v = report.evaluate_shadow({"days_observed": 95, "n": 260, "lcb": 0.04,
                                    "mean": 0.14, "win_rate": 41.5, "brier_skill": 0.015},
                                   frozen_mean=0.60)
        self.assertFalse(v["pass"])

    def test_brier_skill_is_positive_only_for_informative_probabilities(self):
        rng = np.random.default_rng(1)
        won = rng.random(500) < 0.4
        informative = np.where(won, 60.0, 30.0)
        useless = np.full(500, 40.0)
        self.assertGreater(report.brier_skill(informative, won), 0.1)
        self.assertLessEqual(report.brier_skill(useless, won), 0.01)


class TestnetStageTests(unittest.TestCase):
    def test_adverse_selection_blocks_the_testnet_stage(self):
        ex = {"filled": 150, "fill_rate": 0.7,
              "adverse_selection": {"gap_expired_minus_filled_r": 0.25}}
        v = report.evaluate_testnet(ex, days=70, sim_vs_exchange_gap=0.05)
        self.assertFalse(v["pass"])
        self.assertTrue(any("منقضی" in r for r in v["reasons"]))

    def test_healthy_testnet_passes(self):
        ex = {"filled": 150, "fill_rate": 0.7,
              "adverse_selection": {"gap_expired_minus_filled_r": 0.02}}
        self.assertTrue(report.evaluate_testnet(ex, days=70, sim_vs_exchange_gap=0.08)["pass"])

    def test_sim_vs_exchange_gap(self):
        closed = [{"mode": "testnet", "gross_pnl_pct": 1.0, "size_usdt": 1000, "pnl_usdt": 9.0}
                  for _ in range(12)]
        gap = report.sim_vs_exchange_gap(closed)
        self.assertAlmostEqual(gap, 0.1, places=3)
        self.assertIsNone(report.sim_vs_exchange_gap(closed[:5]))


class LiveScaleDecisionTests(unittest.TestCase):
    def _trades(self, n, edge, start, rng):
        r = np.where(rng.random(n) < 0.643 - edge / 2.8, -1.0, 1.8)
        return [{"closed_at_ts": start + k * DAY, "r": float(r[k])} for k in range(n)]

    def test_no_step_up_before_sixty_days_and_trades(self):
        rng = np.random.default_rng(3)
        now = time.time()
        tr = self._trades(30, 0.6, now - 30 * DAY, rng)
        d = report.live_scale_decision(tr, 0.25, now=now)
        self.assertEqual(d["action"], "hold")

    def test_proven_live_edge_steps_up_to_half_percent(self):
        rng = np.random.default_rng(5)
        now = time.time()
        tr = self._trades(90, 0.9, now - 90 * DAY, rng)
        d = report.live_scale_decision(tr, 0.25, now=now)
        self.assertEqual(d["action"], "step_up", d)
        self.assertEqual(d["risk_pct"], 0.50)

    def test_bad_month_demotes_back_to_base_risk(self):
        now = time.time()
        good = [{"closed_at_ts": now - (90 - k) * DAY, "r": 1.8 if k % 2 else -1.0}
                for k in range(60)]
        bad = [{"closed_at_ts": now - (29 - k) * DAY, "r": -1.0} for k in range(25)]
        d = report.live_scale_decision(good + bad, 0.50, now=now)
        self.assertEqual(d["action"], "demote")
        self.assertEqual(d["risk_pct"], 0.25)

    def test_only_the_scale_decision_can_change_live_risk(self):
        tmp = tempfile.TemporaryDirectory()
        old = gates.GATES_PATH
        try:
            gates.GATES_PATH = os.path.join(tmp.name, "g.json")
            with open(gates.GATES_PATH, "w", encoding="utf-8") as f:
                json.dump(dict(gates.DEFAULTS), f)
            gates.load_gates(force=True)
            with self.assertRaises(gates.GateError):
                gates.set_live_risk(1.0, "manual")
            ver = gates.load_gates(force=True)["version"]
            gates.set_live_risk(0.5, "scale", report_token=f"scale:{ver}")
            self.assertEqual(gates.risk_caps()["max_risk_pct_per_trade"], 0.5)
        finally:
            gates.GATES_PATH = old
            gates._cache.update(mtime=None, data=None)
            tmp.cleanup()


class FullReportTests(_IsolatedData, unittest.TestCase):
    def test_report_without_any_proven_combo_is_not_ready_for_live(self):
        rep = report.build(closed_positions=[])
        self.assertFalse(rep["ready_for_live"])
        self.assertFalse(rep["stages"]["frozen_test"]["pass"])
        path = report.write(rep)
        self.assertTrue(os.path.exists(path))

    def test_gate_change_during_proving_resets_the_clock(self):
        research.register({"4h": (0, 10_000_000_000)}, now=time.time() - 100 * DAY)
        gver = gates.load_gates(force=True)["version"]
        gates.set_allowed_combos(["4h|zx|long"], "t", judge_token=f"judge:{gver}",
                                 symbols=["BTCUSDT"])
        rep = report.build(closed_positions=[])
        self.assertGreater(rep["gate_changes_during_proving"], 0)
        self.assertTrue(any("ساعت" in r for r in rep["stages"]["shadow"]["reasons"]))


if __name__ == "__main__":
    unittest.main()
