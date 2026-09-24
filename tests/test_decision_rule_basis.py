# -*- coding: utf-8 -*-
"""جدولِ تصمیم و دفترِ رو-به-جلو روی **قاعده** می‌مانند (calib-F6 / LP-3) و منبعِ داده برچسب دارد (RULE-3).

سیاستِ انتخابِ عملِ «معتبر» در analyze جهت را تحمیل می‌کند؛ این‌جا بررسی می‌شود که تصمیمِ جدول، ترازها
و ردیفِ دفتر عیناً همان تصمیمِ بی‌سیاست بمانند و نظرِ سیاست فقط در ``policy_opinion`` بیاید، و
کارنامهٔ زنده ردیف‌های قدیمیِ سیاست را با قاعده نیامیزد.
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

import decision  # noqa: E402
import engine  # noqa: E402
import gates  # noqa: E402

H_MS = 3_600_000
T0 = 472_222 * H_MS            # مرزِ ساعتِ UTC


def synth(n=1200, seed=3, step_ms=H_MS):
    """گامِ تصادفی با رژیم‌های روند/رنج تا ستاپ‌ها و قاعدهٔ z هر دو رخ دهند (همان test_decision)."""
    rng = np.random.RandomState(seed)
    drift = np.repeat(rng.choice([-1.0, 0.0, 1.0], n // 150 + 1), 150)[:n] * 0.0015
    c = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.0005, n))
    wig = np.abs(rng.normal(0, 0.003, n)) * c
    return {"t": T0 + np.arange(n, dtype=np.int64) * step_ms,
            "o": o, "h": np.maximum(o, c) + wig, "l": np.minimum(o, c) - wig, "c": c,
            "v": rng.lognormal(10, 0.5, n)}


def _lists(kl):
    return {k: np.asarray(v).tolist() for k, v in kl.items()}


def policy_fn(side):
    """سیاستِ «معتبر» و ساختگی که همیشه ``side`` را می‌خواهد (مثلِ خروجیِ calib._score_action_policy)."""
    def fn(tf, dfeats, sfeats, risk_pct, cost=None, regime=None):
        return {"side": side, "source": "action_policy", "policy_trusted": True, "policy_pass": True,
                "policy_score": 1.25, "policy_margin": 1.25, "market_rank_required": True,
                "select_quantile": 0.75, "feature_zmax": 1.0, "n": 120, "avg_r": 0.1,
                "regime_veto": False, "regime_ok": True}
    return fn


class _Gates(unittest.TestCase):
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


KEYS = ("action", "basis", "setup", "entry", "sl", "tp", "rr", "risk_pct", "atr", "grade", "lean",
        "readiness", "scorecard_applies")


class PolicyTakeoverTests(_Gates):
    def test_trusted_policy_cannot_move_the_table_or_the_ledger(self):
        kl = synth(1300, seed=11)
        forced = kept = 0
        for end in range(430, 1300, 29):
            w = _lists({k: v[end - decision.LIVE_BARS:end] for k, v in kl.items()})
            base = engine.analyze(w, "1h")
            want = decision.from_analysis(dict(base, symbol="BTCUSDT"))
            rule_side = base["trade"].get("side")
            # خروجیِ افزوده عیناً تصمیمِ بی‌سیاست است؛ بقیهٔ خروجی دست نخورده
            self.assertEqual(base["rule_trade"]["side"], rule_side, f"end={end}")
            self.assertFalse(base["rule_trade"]["policy_forced"])
            for pside in ("long", "short"):
                a = engine.analyze(w, "1h", action_fn=policy_fn(pside))
                self.assertEqual(a["rule_trade"]["side"], rule_side, f"end={end}")
                d = decision.from_analysis(dict(a, symbol="BTCUSDT"))
                self.assertEqual({k: d[k] for k in KEYS}, {k: want[k] for k in KEYS}, f"end={end} {pside}")
                if a["trade"].get("setup") == "ap":                 # سیاست واقعاً جهت را تحمیل کرد
                    forced += 1
                    self.assertTrue(a["rule_trade"]["policy_forced"])
                    op = d["policy_opinion"]
                    self.assertEqual(op["side"], pside)
                    self.assertEqual(op["same_as_rule"], rule_side == pside)
                    self.assertTrue(op["market_rank_required"])
                    self.assertFalse(op["tradeable"])
                    self.assertIn("رتبهٔ مقطعی", op["note"])
                else:
                    kept += 1                                       # ستاپِ زنده بر سیاست مقدم است
                    self.assertIsNone(d["policy_opinion"])
        self.assertGreater(forced, 10)
        self.assertGreater(kept, 0)

    def test_ledger_rows_under_a_policy_equal_the_rule_rows(self):
        kl = synth(1300, seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for tag, fn in (("rule", None), ("policy", policy_fn("short"))):
                p = paths[tag] = os.path.join(tmp, f"{tag}.jsonl")
                with mock.patch.object(decision, "LEDGER_PATH", p), \
                        mock.patch.object(decision, "VARIANTS_LEDGER_PATH", os.path.join(tmp, f"{tag}.v.jsonl"),
                                          create=True):
                    for end in range(430, 1300, 29):
                        w = {k: v[end - decision.LIVE_BARS:end] for k, v in kl.items()}
                        a = engine.analyze(_lists(w), "1h", action_fn=fn)
                        d = decision.from_analysis(dict(a, symbol="BTCUSDT"))
                        decision.record([d], now=(int(w["t"][-1]) + H_MS) / 1000)

            def rows(p):
                out = []
                for r in decision._read_jsonl(p):
                    r.pop("recorded_at", None)
                    out.append(r)
                return out
            got, want = rows(paths["policy"]), rows(paths["rule"])
            self.assertTrue(want)
            self.assertEqual(got, want)
            self.assertTrue(all(r["basis"] in ("rule", "setup") for r in got))

    def test_record_never_logs_a_policy_basis(self):
        d = {"sym": "BTCUSDT", "tf": "1h", "candle_ts": T0, "action": "long", "basis": "policy",
             "entry": 100.0, "sl": 98.7, "atr": 1.0}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(decision, "LEDGER_PATH", os.path.join(tmp, "l.jsonl")):
            self.assertEqual(decision.record([d], now=(T0 + H_MS) / 1000), 0)
            self.assertEqual(decision.record([dict(d, basis="rule")], now=(T0 + H_MS) / 1000), 1)


class LiveStatsByBasisTests(unittest.TestCase):
    def _open(self, ts, basis, side="long"):
        rid = decision.ledger_id("BTCUSDT", "1h", ts, side)
        return {"kind": "open", "id": rid, "sym": "BTCUSDT", "tf": "1h", "side": side, "candle_ts": ts,
                "entry": 100.0, "sl": 98.7, "atr": 1.0, "basis": basis, "cost_pct": 0.14}

    @staticmethod
    def _res(op, net, exit_ts):
        return {"kind": "result", "id": op["id"], "net_r": net, "exit_ts": exit_ts}

    def test_legacy_policy_rows_are_a_separate_stream(self):
        rule = self._open(T0, "rule")
        pol = self._open(T0 + H_MS, "policy", "short")          # قبلاً با قاعده آمیخته و «هم‌پوشان» بود
        setup = self._open(T0 + 5 * H_MS, "setup")
        old = self._open(T0 + 9 * H_MS, None)                     # ردیفِ بی‌پایه = قاعده
        rows = [rule, pol, setup, old, self._res(rule, 1.5, T0 + 3 * H_MS), self._res(pol, -1.1, T0 + 2 * H_MS),
                self._res(setup, -1.0, T0 + 7 * H_MS), self._res(old, 0.5, T0 + 11 * H_MS)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "l.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(r) for r in rows) + "\n")
            with mock.patch.object(decision, "LEDGER_PATH", path):
                st = decision.live_stats()
                p = decision.payload([decision.from_analysis(
                    {"symbol": "BTCUSDT", "tf": "1h", "zt": T0, "trade": {"side": None}})],
                    scorecard={"cells": {}}, live=st)
        cell = st["cells"]["BTCUSDT"]["1h"]
        self.assertEqual((cell["n"], cell["signals"], cell["overlapping"]), (3, 3, 0))
        self.assertAlmostEqual(cell["avg_r"], round((1.5 - 1.0 + 0.5) / 3, 4))
        self.assertEqual(cell["by_basis"]["rule"]["n"], 2)
        self.assertEqual(cell["by_basis"]["setup"]["n"], 1)
        self.assertEqual(st["overall"]["n"], 3)
        self.assertEqual(st["overall"]["by_basis"]["setup"]["avg_r"], -1.0)
        self.assertEqual(st["policy"]["cells"]["BTCUSDT"]["1h"]["n"], 1)
        self.assertEqual(st["policy"]["overall"]["avg_r"], -1.1)
        self.assertEqual(p["live_basis"], "rule")
        self.assertEqual(p["cells"]["BTCUSDT"]["1h"]["live"]["n"], 3)
        self.assertEqual(p["cells"]["BTCUSDT"]["1h"]["live_policy"]["n"], 1)
        self.assertEqual(p["live_policy_overall"]["n"], 1)
        json.dumps(p, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
