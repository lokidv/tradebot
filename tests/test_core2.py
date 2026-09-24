# -*- coding: utf-8 -*-
"""هستهٔ v2 (core2): برشِ walk-forward بی‌نشت، مدلِ هر ماه فقط ماهِ خودش، قاعدهٔ تصمیم، بازپخشِ برابر با
``decision.replay``، بوت‌استرپِ بلوکِ هفتگی و Holm، و نگهبانِ holdout."""
import importlib.util
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

import core2  # noqa: E402
import core_feats as cf  # noqa: E402
import decision  # noqa: E402

PREREG = os.path.join(ROOT, "bot", "data", "research", "prereg_core2.json")
H_MS = 3_600_000
D_MS = 86_400_000
T0 = 19_700 * D_MS


def synth(n, step_ms=H_MS, seed=3, t0=T0):
    rng = np.random.RandomState(seed)
    drift = np.repeat(rng.choice([-1.0, 0.0, 1.0], n // 150 + 1), 150)[:n] * 0.0015
    c = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.0005, n))
    wig = np.abs(rng.normal(0, 0.003, n)) * c
    v = rng.lognormal(10, 0.5, n)
    return {"t": t0 + np.arange(n, dtype=np.int64) * step_ms, "o": o, "h": np.maximum(o, c) + wig,
            "l": np.minimum(o, c) - wig, "c": c, "v": v, "qv": v * c, "n": np.floor(v / 10) + 1,
            "tbv": v * np.clip(rng.normal(0.5, 0.08, n), 0.05, 0.95)}


def load_cli():
    spec = importlib.util.spec_from_file_location("train_core2_cli", os.path.join(ROOT, "tools", "train_core2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PreregTests(unittest.TestCase):
    def test_constants_match_the_preregistration(self):
        pr = core2.check_prereg(PREREG)
        self.assertEqual(core2.GAP_BARS, 81)
        self.assertEqual(core2.window_ms("validation"), (core2._ms("2024-07-01"), core2._ms("2025-07-01")))
        self.assertIn("0.05", pr["decision_rule_fixed"])

    def test_lgb_params_per_timeframe(self):
        p = core2.lgb_params("5m", 1_000_000)
        self.assertEqual((p["objective"], p["alpha"], p["learning_rate"], p["num_leaves"], p["max_depth"]),
                         ("huber", 1.0, 0.03, 15, 5))
        self.assertEqual(p["min_data_in_leaf"], 2500)
        self.assertEqual(core2.lgb_params("15m", 100_000)["min_data_in_leaf"], 500)
        self.assertEqual(core2.lgb_params("1h", 1)["min_data_in_leaf"], 200)
        self.assertEqual(core2.lgb_params("4h", 1)["min_data_in_leaf"], 100)
        d = core2.lgb_params("1d", 1)
        self.assertEqual((d["num_leaves"], d["min_data_in_leaf"], d["max_depth"]), (7, 60, 5))
        for q in (p, d):
            self.assertEqual((q["feature_fraction"], q["bagging_fraction"], q["bagging_freq"], q["lambda_l2"],
                              q["max_bin"], q["seed"], q["deterministic"]), (0.7, 0.7, 1, 10.0, 63, 42, True))

    def test_equal_total_weight_per_coin(self):
        sym = np.r_[np.zeros(100), np.ones(300), np.full(50, 4)].astype(np.int8)
        w = core2.coin_weights(sym)
        tot = [w[sym == k].sum() for k in (0, 1, 4)]
        self.assertTrue(np.allclose(tot, tot[0]))
        self.assertAlmostEqual(w.mean(), 1.0)


class WalkForwardTests(unittest.TestCase):
    TF = "1h"

    def _ds(self, n=4000, gaps=()):
        """دو ارز روی شبکهٔ ساعتی؛ ``gaps`` = اندیس‌هایی که خروجِ برچسبشان به‌خاطرِ شکافِ داده دیر است."""
        t1 = T0 + np.arange(n, dtype=np.int64) * H_MS
        t = np.r_[t1, t1]
        sym = np.r_[np.zeros(n), np.ones(n)].astype(np.int8)
        exit_close = t + 42 * H_MS
        for g in gaps:
            exit_close[g] = t[g] + 400 * H_MS                  # شکاف: خروج خیلی بعد
        y = np.random.RandomState(0).normal(-0.1, 1, 2 * n).astype(np.float32)
        y[n - 1] = np.nan
        return {"tf": self.TF, "t": t, "sym": sym, "X": np.zeros((2 * n, 3), np.float32), "yL": y, "yS": y.copy(),
                "exit_close": exit_close, "symbols": ["A", "B"], "features": ["f0", "f1", "f2"]}

    def test_cutoff_never_includes_rows_whose_label_exits_at_or_after_it(self):
        ds = self._ds(gaps=(2500, 2600))
        m0 = int(T0 + 2800 * H_MS)
        m = core2.train_mask(ds["t"], ds["exit_close"], ds["yL"], ds["yS"], self.TF, m0)
        self.assertTrue(m.any())
        self.assertTrue((ds["exit_close"][m] <= m0).all())
        self.assertTrue((ds["t"][m] <= m0 - 81 * H_MS).all())
        self.assertFalse(m[2500] or m[2600])                  # شکافِ داده: t زود ولی خروج پس از برش
        self.assertTrue(m[2499])
        self.assertFalse(m[2800 - 80])                        # درونِ ۸۱ کندلِ پاک‌سازی + embargo
        self.assertFalse(m[3999])                             # برچسبِ NaN

    def test_5m_uses_every_second_row(self):
        t = T0 + np.arange(1000, dtype=np.int64) * 300_000
        y = np.zeros(1000, np.float32)
        m = core2.train_mask(t, t + 42 * 300_000, y, y, "5m", int(t[-1]) + 1)
        self.assertTrue(((t[m] // 300_000) % 2 == 0).all())
        self.assertEqual(int(m.sum()), len(np.flatnonzero((t <= t[-1] + 1 - 81 * 300_000) & ((t // 300_000) % 2 == 0))))

    def test_monthly_model_only_predicts_its_own_month(self):
        ds = self._ds(n=4000, gaps=(3000,))
        months = core2.month_starts(int(T0 + 3000 * H_MS), int(T0 + 3900 * H_MS))
        self.assertGreaterEqual(len(months), 2)
        seen = []

        def fit_fn(X, yL, yS, t, sym, tf, names, threads):
            k = len(seen)
            seen.append((int(t.max()), len(t)))
            return (lambda Xte: (np.full(len(Xte), k, np.float32), np.full(len(Xte), -k, np.float32))), {"k": k}

        oos, meta = core2.walk_forward(ds, months, fit_fn=fit_fn, window_end=months[-1][1])
        self.assertEqual(len(seen), len(months))
        for k, (m0, m1, ym) in enumerate(months):
            self.assertLessEqual(seen[k][0], m0 - 81 * H_MS)
            rows = oos["month"] == ym
            self.assertTrue(rows.any())
            self.assertTrue(((oos["t"][rows] >= m0) & (oos["t"][rows] < m1)).all())
            self.assertTrue((oos["E_L"][rows] == k).all())    # فقط مدلِ همان ماه
            exp = int(((ds["t"] >= m0) & (ds["t"] < m1)).sum())
            self.assertEqual(int(rows.sum()), exp)
            self.assertLessEqual(meta[k]["train_exit_close_max"], core2._date(m0))
        self.assertTrue(seen[1][1] > seen[0][1])              # پنجرهٔ آموزشِ رو به گسترش
        self.assertLess(int(oos["t"].max()), months[-1][1])

    def test_window_end_refuses_months_past_it(self):
        ds = self._ds()
        months = core2.month_starts(int(T0 + 3000 * H_MS), int(T0 + 3900 * H_MS))
        with self.assertRaises(ValueError):
            core2.walk_forward(ds, months, fit_fn=lambda *a: None, window_end=months[0][1])

    def test_es_split_keeps_a_41_bar_gap(self):
        t = T0 + np.arange(2000, dtype=np.int64) * H_MS
        tr, va, ts = core2.es_split(np.r_[t, t], "1h")
        self.assertAlmostEqual(va.mean(), 0.15, places=2)
        self.assertEqual(int(np.r_[t, t][tr].max()), ts - 41 * H_MS)
        self.assertFalse((tr & va).any())

    def test_fit_pair_runs_lightgbm(self):
        rng = np.random.RandomState(1)
        n = 900
        X = rng.normal(size=(n, 4)).astype(np.float32)
        y = (0.3 * X[:, 0] + rng.normal(0, 0.5, n)).astype(np.float32)
        t = T0 + np.arange(n, dtype=np.int64) * D_MS
        pred, meta = core2.fit_pair(X, y, -y, t, np.arange(n) % 3, "1d", ["a", "b", "c", "d"], threads=2)
        EL, ES = pred(X[:10])
        self.assertEqual(EL.shape, (10,))
        self.assertGreaterEqual(meta["long"]["best_iter"], 1)
        self.assertEqual(meta["min_data_in_leaf"], 60)


class DatasetTests(unittest.TestCase):
    def test_rows_need_a_futures_bar_and_warmup(self):
        tf = "1d"
        syms = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "TRXUSDT")
        n = 800
        spot = {s: synth(n, D_MS, seed=k) for k, s in enumerate(syms)}
        fut = {s: synth(n, D_MS, seed=k + 50) for k, s in enumerate(syms)}
        drop = 700
        fut["ETHUSDT"] = {k: np.delete(v, drop) for k, v in fut["ETHUSDT"].items()}
        end = int(spot["BTCUSDT"]["t"][-1]) + D_MS
        ds = core2.build_dataset(tf, end, end, symbols=syms, spot=spot, fut=fut, kfund={})
        self.assertEqual(ds["X"].shape[1], len(cf.FEATURES[tf]))
        self.assertEqual(ds["X"].dtype, np.float32)
        eth = ds["sym"] == 1
        self.assertNotIn(int(spot["ETHUSDT"]["t"][drop]), set(ds["t"][eth].tolist()))
        self.assertEqual(int(eth.sum()), n - core2.TRAIN_WARMUP[tf] - 1)
        self.assertEqual(int(ds["t"][ds["sym"] == 0].min()), int(spot["BTCUSDT"]["t"][core2.TRAIN_WARMUP[tf]]))
        ok = ds["exit_close"] < np.iinfo(np.int64).max
        self.assertTrue((ds["exit_close"][ok] > ds["t"][ok]).all())
        self.assertTrue(np.isnan(ds["yL"][~ok]).all() or not (~ok).any())


class DecisionRuleTests(unittest.TestCase):
    def test_fixed_rule(self):
        EL = np.array([0.10, 0.05, 0.06, 0.04, -0.2, 0.30, np.nan, 0.20])
        ES = np.array([0.00, 0.00, 0.02, -0.5, 0.10, 0.28, 0.30, 0.20])
        self.assertEqual(core2.decide(EL, ES).tolist(), [1, 1, 0, 0, -1, 0, 0, 0])

    def test_margin_zero_and_ties_wait(self):
        self.assertEqual(core2.decide([0.0, 0.1, -0.01], [0.0, 0.05, -0.02], 0.0).tolist(), [0, 1, 0])
        self.assertEqual(core2.decide([0.05], [0.3], 0.2).tolist(), [-1])
        self.assertEqual(core2.decide([0.1], [0.29], 0.2).tolist(), [0])


class ReplayEqualityTests(unittest.TestCase):
    def test_same_side_series_gives_the_same_trades_as_decision_replay(self):
        kl = synth(1400)
        kl = {k: kl[k] for k in ("t", "o", "h", "l", "c", "v")}
        rep = decision.replay(kl, first=600)
        self.assertGreater(len(rep["trades"]), 5)
        mine = core2.replay_sides(kl, rep["side"], rep["first"])
        keys = ("i", "t", "side", "entry", "sl", "tp", "risk_pct", "outcome", "gross_r", "net_r", "exit_idx", "bars_held")
        self.assertEqual([tuple(x[k] for k in keys) for x in mine], [tuple(x[k] for k in keys) for x in rep["trades"]])

    def test_one_trade_at_a_time_and_next_open_entry(self):
        kl = synth(900)
        side = np.zeros(900, np.int8)
        side[500:600] = 1
        tr = core2.replay_sides(kl, side, 450)
        self.assertTrue(tr)
        for a, b in zip(tr, tr[1:]):
            self.assertGreaterEqual(b["i"], a["exit_idx"])
        self.assertTrue(all(x["entry"] == kl["o"][x["i"] + 1] for x in tr))

    def test_evaluate_runs_rule_and_v2_on_the_same_bars(self):
        syms = ("A", "B")
        fut = {s: synth(3000, seed=k + 9) for k, s in enumerate(syms)}
        t = fut["A"]["t"]
        w0, w1 = int(t[2400]), int(t[2900])
        rng = np.random.RandomState(0)
        ot = np.r_[t[2400:2900], t[2400:2900]]
        oos = {"sym": np.r_[np.zeros(500), np.ones(500)].astype(np.int8), "t": ot,
               "E_L": rng.normal(0, 0.1, 1000).astype(np.float32), "E_S": rng.normal(0, 0.1, 1000).astype(np.float32)}
        v2, rule, same = core2.evaluate("1h", oos, w0, w1, fut, symbols=syms)
        self.assertTrue(same)
        for s in syms:
            self.assertTrue(v2[core2.MARGIN][s])
            self.assertTrue(all(w0 <= x["t"] < w1 for x in v2[core2.MARGIN][s] + rule[s]))
        n = {m: sum(len(v2[m][s]) for s in syms) for m in v2}
        self.assertGreaterEqual(n[0.0], n[0.2])


class StatsTests(unittest.TestCase):
    W0, W1 = core2._ms("2024-07-01"), core2._ms("2025-07-01")

    def test_week_blocks_start_monday(self):
        mon = core2._ms("2024-07-01")                          # دوشنبه
        self.assertEqual(core2.week_id([mon])[0], core2.week_id([mon + 7 * D_MS - 1])[0])
        self.assertNotEqual(core2.week_id([mon - 1])[0], core2.week_id([mon])[0])

    def _trades(self, mu, n, seed):
        rng = np.random.RandomState(seed)
        return ((self.W0 + (rng.rand(n) * (self.W1 - self.W0)).astype(np.int64)), rng.normal(mu, 1.0, n))

    def test_bootstrap_is_seeded_and_sensible(self):
        ta, ra = self._trades(0.4, 600, 1)
        tb, rb = self._trades(0.0, 600, 2)
        a = core2.block_bootstrap(ta, ra, tb, rb, self.W0, self.W1)
        b = core2.block_bootstrap(ta, ra, tb, rb, self.W0, self.W1)
        self.assertEqual(a, b)
        self.assertEqual(a["weeks"], 53)
        self.assertLess(a["ci95"][0], a["mean"])
        self.assertGreater(a["ci95"][1], a["mean"])
        self.assertGreater(a["diff_ci95"][0], 0)
        self.assertLess(a["p_one_sided"], 0.01)
        neg = core2.block_bootstrap(tb, rb - 0.3, ta, ra, self.W0, self.W1)
        self.assertGreater(neg["p_one_sided"], 0.9)
        self.assertLess(neg["diff_ci95"][1], 0)

    def test_bootstrap_with_no_v2_trades(self):
        tb, rb = self._trades(0.0, 50, 2)
        r = core2.block_bootstrap(np.zeros(0, np.int64), np.zeros(0), tb, rb, self.W0, self.W1)
        self.assertIsNone(r["mean"])
        self.assertEqual(r["p_one_sided"], 1.0)

    def test_bootstrap_rejects_trades_outside_the_window(self):
        with self.assertRaises(ValueError):
            core2.block_bootstrap([self.W1 + 8 * D_MS], [1.0], [], [], self.W0, self.W1)

    def test_holm(self):
        h = core2.holm({"5m": 0.01, "15m": 0.04, "1h": 0.03, "4h": 0.2, "1d": None}, 0.10)
        self.assertAlmostEqual(h["5m"][0], 0.05)
        self.assertAlmostEqual(h["1h"][0], 0.12)
        self.assertAlmostEqual(h["15m"][0], 0.12)
        self.assertAlmostEqual(h["4h"][0], 0.4)
        self.assertEqual(h["1d"], (1.0, False))
        self.assertEqual([k for k, v in h.items() if v[1]], ["5m"])

    def _res(self, mean=0.2, n=150, rule=0.0, lo=0.05, coins=(0.3, 0.2, 0.1, -0.1, 0.1)):
        tot = sum(coins)
        return {"v2": {"mean": mean, "n": n, "coins_positive": sum(c > 0 for c in coins),
                       "max_coin_share": max(coins) / tot},
                "rule": {"mean": rule}, "bootstrap": {"diff_ci95": [lo, 0.4]}}

    def test_adopt_rule(self):
        self.assertTrue(core2.adopt_checks("1h", self._res(), True)["adopted"])
        self.assertFalse(core2.adopt_checks("1h", self._res(), False)["adopted"])        # Holm
        self.assertFalse(core2.adopt_checks("1h", self._res(n=99), True)["adopted"])     # n
        self.assertTrue(core2.adopt_checks("1d", self._res(n=40), True)["adopted"])      # 1d: n ≥ 40
        self.assertFalse(core2.adopt_checks("1h", self._res(rule=0.16), True)["adopted"])  # +0.05R
        self.assertFalse(core2.adopt_checks("1h", self._res(lo=-0.01), True)["adopted"])   # CI تفاوت
        self.assertFalse(core2.adopt_checks("1h", self._res(coins=(0.3, 0.2, -0.1, -0.1, -0.1)), True)["adopted"])
        self.assertFalse(core2.adopt_checks("1h", self._res(coins=(0.9, 0.1, 0.1, 0.1, -0.1)), True)["adopted"])

    def test_trade_stats_shares(self):
        tb = {"A": [{"t": 1, "gross_r": 1.0, "risk_pct": 1.0, "side": "long"}],
              "B": [{"t": 2, "gross_r": -0.5, "risk_pct": 1.0, "side": "short"}]}
        st = core2.trade_stats(core2.trade_array(tb, ["A", "B"], 0.0), ["A", "B"])
        self.assertEqual((st["n"], st["coins_positive"], st["long_n"], st["short_n"]), (2, 1, 1, 1))
        self.assertAlmostEqual(st["max_coin_share"], 2.0)


class HoldoutGuardTests(unittest.TestCase):
    def _report(self, adopted=True, complete=True):
        return {"window": "validation", "complete": complete,
                "timeframes": {tf: {"adopted": adopted and tf == "4h"} for tf in core2.TFS}}

    def test_guard(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(core2.holdout_guard("4h", None, d)[0])
            self.assertFalse(core2.holdout_guard("4h", self._report(complete=False), d)[0])
            self.assertFalse(core2.holdout_guard("1h", self._report(), d)[0])
            self.assertFalse(core2.holdout_guard("4h", self._report(adopted=False), d)[0])
            self.assertTrue(core2.holdout_guard("4h", self._report(), d)[0])
            with open(core2.result_path("4h", "holdout", d), "w") as f:
                f.write("{}")
            self.assertFalse(core2.holdout_guard("4h", self._report(), d)[0])   # یک‌بارمصرف

    def test_run_window_refuses_holdout(self):
        with mock.patch.object(core2, "build_dataset", side_effect=AssertionError("نباید داده بخواند")):
            with self.assertRaises(PermissionError):
                core2.run_window("4h", "holdout")
            with self.assertRaises(PermissionError):
                core2.run_window("4h", "holdout", allow_holdout=True)            # گزارشی نیست

    def test_cli_refuses_without_flag_and_without_adoption(self):
        cli = load_cli()
        with mock.patch.object(core2, "check_prereg"), \
                mock.patch.object(core2, "run_window", side_effect=AssertionError("نباید اجرا شود")), \
                mock.patch("sys.stderr"):
            self.assertEqual(cli.main(["--tf", "4h", "--window", "holdout"]), 2)
            self.assertEqual(cli.main(["--tf", "4h", "--holdout"]), 2)
            self.assertEqual(cli.main(["--tf", "4h", "--window", "holdout", "--holdout"]), 2)
            os.makedirs(core2.RESEARCH_DIR, exist_ok=True)
            path = os.path.join(core2.RESEARCH_DIR, core2.REPORT_PREFIX + "20990101_000000.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._report(adopted=False), f)
            try:
                self.assertEqual(cli.main(["--tf", "4h", "--window", "holdout", "--holdout"]), 2)
            finally:
                os.remove(path)

    def test_core2_never_touches_gates_or_tradeable(self):
        with open(os.path.join(ROOT, "bot", "core2.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("import gates", src)
        self.assertNotIn("gates.json\"", src)
        self.assertNotIn("tradeable =", src)
        self.assertNotIn("tradeable\"] =", src)


if __name__ == "__main__":
    unittest.main()
